import abc
import torch
import torch.nn.functional as F
from torch import nn
from typing import Optional, List
from diffusers.models.attention_processor import Attention
from diffusers.models.attention_processor import AttnProcessor, AttnProcessor2_0
from diffusers.utils.constants import USE_PEFT_BACKEND
from matplotlib import pyplot as plt
from torch import einsum
from einops import rearrange

class Hack_AttnProcessor:

    def __init__(self, attnstore, layer_in_dit):
        super().__init__()
        self.attnstore = attnstore
        self.layer_in_dit = layer_in_dit

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        scale: float = 1.0,
    ) -> torch.Tensor:
        residual = hidden_states

        args = () if USE_PEFT_BACKEND else (scale,)

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length,batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        # query = attn.to_q(hidden_states)
        query = attn.to_q(hidden_states, *args)

        is_cross = encoder_hidden_states is not None
        # encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        
        key = attn.to_k(encoder_hidden_states, *args)
        value = attn.to_v(encoder_hidden_states, *args)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        # import pdb
        # pdb.set_trace()

        # print(f"query:{query.size()}")
        # print(f"key:{key.size()}")
        '''
        query:torch.Size([32, 1024, 72])
        key:torch.Size([32, 1024, 72])
        '''
        # sim = einsum('b i d, b j d -> b i j', query, key) * scale
        # self_attn_map = sim.softmax(dim=-1)
        # self_attn_map = rearrange(self_attn_map, 'h n m -> n (h m)')

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        self.attnstore(attention_probs, is_cross, self.layer_in_dit)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
    
def register_attention_control(model, controller):
    # indices = [int(item.split('.')[1]) for item in processor_list]

    attn_procs = {}
    cross_att_count = 0

    # print(model.transformer.attn_processors.keys())
    '''
    dict_keys(['transformer_blocks.0.attn1.processor', 'transformer_blocks.0.attn2.processor', 'transformer_blocks.1.attn1.processor', 'transformer_blocks.1.attn2.processor', 'transformer_blocks.2.attn1.processor', 'transformer_blocks.2.attn2.processor', 'transformer_blocks.3.attn1.processor', 'transformer_blocks.3.attn2.processor', 'transformer_blocks.4.attn1.processor', 'transformer_blocks.4.attn2.processor', 'transformer_blocks.5.attn1.processor', 'transformer_blocks.5.attn2.processor', 'transformer_blocks.6.attn1.processor', 'transformer_blocks.6.attn2.processor', 'transformer_blocks.7.attn1.processor', 'transformer_blocks.7.attn2.processor', 'transformer_blocks.8.attn1.processor', 'transformer_blocks.8.attn2.processor', 'transformer_blocks.9.attn1.processor', 'transformer_blocks.9.attn2.processor', 'transformer_blocks.10.attn1.processor', 'transformer_blocks.10.attn2.processor', 'transformer_blocks.11.attn1.processor', 'transformer_blocks.11.attn2.processor', 'transformer_blocks.12.attn1.processor', 'transformer_blocks.12.attn2.processor', 'transformer_blocks.13.attn1.processor', 'transformer_blocks.13.attn2.processor', 'transformer_blocks.14.attn1.processor', 'transformer_blocks.14.attn2.processor', 'transformer_blocks.15.attn1.processor', 'transformer_blocks.15.attn2.processor', 'transformer_blocks.16.attn1.processor', 'transformer_blocks.16.attn2.processor', 'transformer_blocks.17.attn1.processor', 'transformer_blocks.17.attn2.processor', 'transformer_blocks.18.attn1.processor', 'transformer_blocks.18.attn2.processor', 'transformer_blocks.19.attn1.processor', 'transformer_blocks.19.attn2.processor', 'transformer_blocks.20.attn1.processor', 'transformer_blocks.20.attn2.processor', 'transformer_blocks.21.attn1.processor', 'transformer_blocks.21.attn2.processor', 'transformer_blocks.22.attn1.processor', 'transformer_blocks.22.attn2.processor', 'transformer_blocks.23.attn1.processor', 'transformer_blocks.23.attn2.processor', 'transformer_blocks.24.attn1.processor', 'transformer_blocks.24.attn2.processor', 'transformer_blocks.25.attn1.processor', 'transformer_blocks.25.attn2.processor', 'transformer_blocks.26.attn1.processor', 'transformer_blocks.26.attn2.processor', 'transformer_blocks.27.attn1.processor', 'transformer_blocks.27.attn2.processor'])
    '''

    for name in model.transformer.attn_processors.keys():
        
        # import pdb
        # pdb.set_trace()
        if 'fuser' in name: continue
        
        layer_in_dit = int(name.split('.')[1])

        if 'attn1' in name:
            cross_att_count += 1
            attn_procs[name] = Hack_AttnProcessor(
                attnstore=controller, layer_in_dit=layer_in_dit
            )
        else:
            attn_procs[name] = AttnProcessor2_0()

    # set_attn_processor需要实现
    # import pdb
    # pdb.set_trace()
    model.transformer.set_attn_processor(attn_procs)
    controller.num_att_layers = cross_att_count
    


class AttentionControl(abc.ABC):

    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    @property
    def num_uncond_att_layers(self):
        return 0

    @abc.abstractmethod
    def forward(self, attn, is_cross: bool, layer_in_dit: int):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, layer_in_dit: int):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            # conditional attention
            # h = attn.shape[0]
            # self[h//2:].forward(attn[h//2:], is_cross, layer_in_dit)
            attn = self.forward(attn, is_cross, layer_in_dit)

        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0


class EmptyControl(AttentionControl):

    def forward(self, attn, is_cross: bool, layer_in_dit: int):
        return attn


class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {
            "0_self": [], "0_cross": [],
            "1_self": [], "1_cross": [],
            "2_self": [], "2_cross": [],
            "3_self": [], "3_cross": [],
            "4_self": [], "4_cross": [],
            "5_self": [], "5_cross": [],
            "6_self": [], "6_cross": [],
            "7_self": [], "7_cross": [],
            "8_self": [], "8_cross": [],
            "9_self": [], "9_cross": [],
            "10_self": [], "10_cross": [],
            "11_self": [], "11_cross": [],
            "12_self": [], "12_cross": [],
            "13_self": [], "13_cross": [],
            "14_self": [], "14_cross": [],
            "15_self": [], "15_cross": [],
            "16_self": [], "16_cross": [],
            "17_self": [], "17_cross": [],
            "18_self": [], "18_cross": [],
            "19_self": [], "19_cross": [],
            "20_self": [], "20_cross": [],
            "21_self": [], "21_cross": [],
            "22_self": [], "22_cross": [],
            "23_self": [], "23_cross": [],
            "24_self": [], "24_cross": [],
            "25_self": [], "25_cross": [],
            "26_self": [], "26_cross": [],
            "27_self": [], "27_cross": []
        }

        # return {"down_cross": [], "mid_cross": [], "up_cross": [],
        #         "down_self": [], "mid_self": [], "up_self": []}

    def forward(self, attn, is_cross: bool, layer_in_dit: int):
        key = f"{layer_in_dit}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 32 ** 2:  # avoid memory overhead
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        self.attention_store = self.step_store
        if self.save_global_store:
            with torch.no_grad():
                if len(self.global_store) == 0:
                    self.global_store = self.step_store
                else:
                    for key in self.global_store:
                        for i in range(len(self.global_store[key])):
                            self.global_store[key][i] += self.step_store[key][i].detach()
        self.step_store = self.get_empty_store()
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = self.attention_store
        return average_attention

    def get_average_global_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.global_store[key]] for key in
                             self.attention_store}
        return average_attention

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.global_store = {}

    def __init__(self, save_global_store=False):
        '''
        Initialize an empty AttentionStore
        :param step_index: used to visualize only a specific step in the diffusion process
        '''
        super(AttentionStore, self).__init__()
        self.save_global_store = save_global_store
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.global_store = {}
        self.curr_step_index = 0


def get_self_attention_map(attention_store: AttentionStore,
                           tgt_res: int,
                           from_which_layer: int,
                           is_cross: bool,
                           ):
    from_which_layer = f"{str(from_which_layer)}_{'cross' if is_cross else 'self'}"
    # import pdb
    # pdb.set_trace()
    attn_map = attention_store.attention_store[from_which_layer][0]
    # import pdb
    # pdb.set_trace()
    # for conditional score attention map #
    h = attn_map.shape[0]
    attn_map = attn_map[h//2:]  # CFG torch.Size([16, 1024, 1024])
    # for conditional score attention map #
    self_attn_map = rearrange(attn_map, 'h n m -> n (h m)')
    

    # attn_map = attn_map.mean(dim=0)
    # attn_map = torch.nn.functional.interpolate(attn_map.unsqueeze(0).unsqueeze(0).cuda(),tgt_res,mode='bilinear').cpu()
    # attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())
    # attn_map = attn_map.squeeze(0).squeeze(0).numpy()

    return self_attn_map 

def save_attention_map_as_image(attn_map, save_path, title=None, cmap='hot'):
    attn_map_np = attn_map.cpu().numpy() if torch.is_tensor(attn_map) else attn_map
    fig, ax = plt.subplots()
    
    cax = ax.imshow(attn_map_np, cmap=cmap, interpolation='nearest')
    fig.colorbar(cax) 
    
    if title:
        ax.set_title(title)
    ax.axis('off')  

    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)


def aggregate_attention(attention_store: AttentionStore,
                        res: int,
                        from_where: List[str],
                        is_cross: bool,
                        select: int) -> torch.Tensor:
    """ Aggregates the attention across the different layers and heads at the specified resolution. """
    out = []
    attention_maps = attention_store.get_average_attention()
    num_pixels = res ** 2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(1, -1, res, res, item.shape[-1])[select]
                out.append(cross_maps)
    out = torch.cat(out, dim=0)
    out = out.sum(0) / out.shape[0]
    return out


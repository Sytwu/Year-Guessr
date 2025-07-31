import os.path as osp

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import torchvision.models as models
import clip as CLIP
from clip import clip

from ordinalclip.utils import get_logger

from . import image_encoders
from .builder import MODELS
from .prompt_leaners import PROMPT_LEARNERS
from .prompt_leaners.plain_prompt_learner import PlainPromptLearner

from .location_encoder import LocationEncoder

import sys

logger = get_logger(__name__)


# for built year estimation
bin_list_a  = [800, 1150, 1400, 1600, 1800, 1900, 1950] 
bin_list_b  = [800, 1150, 1400, 1600, 1800, 1900, 1950] 

bin_width_a = [350,  250,  200,  200,  100,   50,   75] 
bin_width_b = [350,  250,  200,  200,  100,   50,   75]

# bin_list_a  = [800, 1150, 1400, 1600, 1800, 1900, 1950] 
# bin_list_b  = [350,  250,  200,  200,  100,   50,   75] 
# 
# bin_width_a = [350,  250,  200,  200,  100,   50,   75] 
# bin_width_b = [350,  250,  200,  200,  100,   50,   75]  

# for age estimation
# bin_list_a = [0, 13, 19, 35, 65] 
# bin_list_b = [0, 13, 19, 35, 65] 

# bin_width_a = [13,6,16,30,36]
# bin_width_b = [13,6,16,30,36]


# for image aesthetics
# bin_list_a = [0, 1, 2, 3, 4] 
# bin_list_b = [0, 1, 2, 3, 4] 

# bin_width_a = [1, 1, 1, 1, 1] 
# bin_width_b = [1, 1, 1, 1, 1] 


# for historical image dating
# bin_list_a = [0, 1, 2, 3, 4] 
# bin_list_b = [0, 1, 2, 3, 4] 

# bin_width_a = [1, 1, 1, 1, 1] 
# bin_width_b = [1, 1, 1, 1, 1] 

@MODELS.register_module()
class RegCLIPSSR(nn.Module):
    def __init__(
        self,
        text_encoder_name,
        image_encoder_name,
        prompt_learner_cfg,
        d = 512,
        **kwargs,
    ) -> None:
        super().__init__()

        if kwargs:
            logger.info(f"irrelevant kwargs: {kwargs}")

        clip_model = load_clip_to_cpu(
            text_encoder_name,
            image_encoder_name,
            root=osp.join(osp.dirname(osp.realpath(__file__)), "..", "..", ".cache", "clip"),
        )
        clip_model.float()
        logger.info("convert `clip_model` to float32. if need fp16 model, call `clip.model.convert_weights`")

        self.image_encoder = clip_model.visual
        self.location_encoder = LocationEncoder(from_pretrained=True)
        self.text_encoder = TextEncoder(clip_model)
        prompt_learner_cfg.update(dict(clip_model=clip_model))
        self.prompt_learner: PlainPromptLearner = PROMPT_LEARNERS.build(prompt_learner_cfg)
        self.psudo_sentence_tokens = self.prompt_learner.psudo_sentence_tokens
        self.logit_scale = clip_model.logit_scale

        self.embed_dims = clip_model.text_projection.shape[1]
        self.num_ranks = self.prompt_learner.num_ranks
        self.d = d
        self.zero_conv = nn.Linear(self.d, self.d)
        self.zero_conv.weight.data.fill_(0)
        self.zero_conv.bias.data.fill_(0)

        # we first adopt CLIP-adapter based adaptation method. After experiment, we found fully finetune the image encoder could get the better performance.
        self.image_adapter = Adapter(self.d, 4)
        
        # YearCLIP
        opts = [
            [  # Roof Style
                'spire (a sharply pointed roof that emphasizes verticality and ornate detailing)',
                'dome (a smoothly curved roof suggesting grandeur and centrality)',
                'flat roof (a completely horizontal surface with an unobstructed and minimalist design)',
                'sloped roof (a roof with a noticeable and functional inclination for water drainage and dynamic appearance)',
                'gabled roof (a traditional peaked roof with a triangular profile that exudes symmetry)',
                'mansard roof (a dual-pitched roof offering both elegance and additional living space)',
                'butterfly roof (an inverted roof design that creates a V-shaped, modern and unconventional look)',
            ],
            [  # Window Style
                'arched window (featuring a curved top that adds classical elegance)',
                'rectangular window (a clean, box-like shape emphasizing simplicity)',
                'bay window (a protruding window design creating depth and character)',
                'curtain window (a large fixed-pane window providing expansive views and modern aesthetics)',
                'casement window (a hinged window that swings outward for both function and style)',
                'sash window (a vertically sliding window evoking traditional craftsmanship)',
            ],
            [  # Wall Material
                'brick wall (made of red or brown bricks suggesting historical construction)',
                'stone wall (constructed from rough-hewn stone indicating traditional masonry)',
                'concrete wall (a smooth or textured surface made of reinforced concrete)',
                'glass curtain wall (a sleek, transparent facade emblematic of modernist design)',
                'stucco wall (a fine plastered exterior that conveys classical aesthetics)',
                'timber wall (wooden elements showcasing warmth and organic texture)',
                'metal cladding wall (modern metallic surface offering an industrial look)',
            ],
            [  # Decorative Style
                'Gothic style (characterized by pointed arches and intricate stonework)',
                'Renaissance style (marked by proportion, symmetry, and classical elements)',
                'Baroque style (lavish ornamentation with curves and bold details)',
                'Neoclassical style (based on Greek and Roman simplicity and columns)',
                'Art Deco style (bold lines, symmetry, and metallic finishes)',
                'Postmodern style (eclectic forms with irony and historical references)',
                'Modernist style (minimalist, functional, and clean geometric forms)',
                'Minimalist style (emphasizing essential shapes and reducing decoration)',
            ],
            [  # Building Height
                'low-rise (a building with 1 to 3 floors, often residential or early commercial)',
                'mid-rise (a building with 4 to 7 floors, common in post-war development)',
                'high-rise (a tall structure with more than 7 floors, typically modern or contemporary)',
                'skyscraper (a very tall building with steel-frame construction and glass facades)',
            ],
            [  # Number of Floors
                'one-story (a single-level building, typical of early or suburban structures)',
                'two-story (a common residential layout with moderate height)',
                'multi-story (three or more levels, indicating urbanization or modern design)',
            ],
            [  # Ornamentation Level
                'highly ornate (richly detailed façades, carvings, and embellishments)',
                'moderately ornate (some decorative elements balanced with clean lines)',
                'plain (very minimal or no visible ornamentation)',
            ],
            [  # Color Scheme
                'earth tones (browns, beiges, and warm colors associated with tradition)',
                'monochrome (mostly black, white, or gray, often modern in style)',
                'pastels (light, soft colors common in mid-century or regional styles)',
                'bold colors (bright, vivid hues suggesting postmodern or eclectic design)',
            ],
            [  # Structural Shape
                'rectangular layout (box-like structure typical in utilitarian or early styles)',
                'L-shaped layout (an L configuration providing functional separation)',
                'U-shaped layout (forms an open courtyard, common in institutional buildings)',
                'irregular layout (asymmetric or unusual geometry, often seen in avant-garde architecture)',
            ],
            [  # Entrance Design
                'arched entrance (a rounded entryway reflecting classical design)',
                'grand staircase entrance (a prominent staircase leading to the main door)',
                'recessed entrance (a door set back into the building, offering subtlety)',
                'glass door entrance (modern and transparent, emphasizing openness)',
            ],
            [  # Balcony Presence
                'no balcony (a flat facade without extensions)',
                'small balconies (individual units with modest outdoor spaces)',
                'wraparound balconies (large, continuous balcony around the building)',
            ],
            [  # Column Presence
                'no columns (a flat facade with no structural or decorative columns)',
                'decorative columns (stylized columns for aesthetic effect)',
                'load-bearing columns (visible structural supports)',
            ],
            [  # Façade Symmetry
                'symmetrical facade (balanced and mirrored on both sides)',
                'asymmetrical facade (irregular or creative placement of windows and doors)',
            ],
            [  # Roof Material
                'clay tiles (red or orange traditional tiles seen in Mediterranean or colonial styles)',
                'slate (dark, flat stones common in Gothic or European buildings)',
                'metal roofing (shiny or matte modern surface for durability)',
                'asphalt shingles (common in suburban and American residential styles)',
            ],
            [  # Construction Period Clues
                'features typical of pre-war architecture (intricate masonry, high ornamentation)',
                'features typical of mid-century modern (flat planes, large glass windows)',
                'features typical of postmodern architecture (irony, mixed materials, bold forms)',
                'features typical of contemporary architecture (minimalism, tech integration)',
            ],
        ]

        reasoning = [
            'The roof of the building in the image is a ',
            'The window of the building in the image is a ',
            'The wall of the building in the image is made of ',
            'The building features a decorative style of ',
            'The height classification of the building is ',
            'The number of floors in the building is ',
            'The level of ornamentation is ',
            'The building’s color scheme is ',
            'The structural shape of the building is ',
            'The entrance of the building has a design of ',
            'The balcony configuration of the building is ',
            'The building has ',
            'The building has a façade that is ',
            'The material used for the roof is ',
            'The architectural period this building resembles is ',
        ]

        reasoning_prompts = [
            [prefix + option for option in group]
            for prefix, group in zip(reasoning, opts)
        ]
        
        self.reasoning_prompts = [
            p for group in reasoning_prompts for p in group
        ]
        ex_stage_num = len(self.reasoning_prompts)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = CLIP.load("ViT-B/32", device=device)
        tokenized = clip.tokenize(self.reasoning_prompts).to(device)
        
        with torch.no_grad():
            feats = model.encode_text(tokenized)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        self.reasoning_features = feats

        self.regressor = SSRModule(stage_num=[7, 3], ex_stage_num=ex_stage_num, class_range=1025)

    def forward(self, images, location, run_type, show_importance=False):
        sentence_embeds = self.prompt_learner()
        psudo_sentence_tokens = self.psudo_sentence_tokens
        text_features = self.text_encoder(sentence_embeds, psudo_sentence_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        image_features = self.image_encoder(images)
        """
        if location:
            location_features = self.location_encoder(location)
        else:
            location_features = self.geoclip_image_encoder(images)
        zero_image = self.zero_conv(image_features)
        image_features = location_features + zero_image
        """

        y = self.image_adapter(image_features)
        y_ratio = 0.8
        image_features = y_ratio * y + (1 - y_ratio) * image_features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()
        
        if location:
            location_features = self.location_encoder(location)
            location_features = self.zero_conv(location_features)
            image_features = image_features + location_features
        
        # print(f"Text = {text_features.shape}")
        # print(f"Reason = {self.reasoning_features.shape}")
        reasoning_logits = logit_scale * image_features @ self.reasoning_features.t()

        regress_age = self.regressor(logits, reasoning_logits, show_importance)

        return logits, regress_age, image_features, text_features

    def forward_text_only(self):
        sentence_embeds = self.prompt_learner()
        psudo_sentence_tokens = self.psudo_sentence_tokens
        text_features = self.text_encoder(sentence_embeds, psudo_sentence_tokens)

        return text_features

    def encode_image(self, x):
        return self.image_encoder(x)


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection

    def forward(self, prompts, tokenized_prompts):
        x = prompts.type(self.dtype) + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

    @property
    def dtype(self):
        return self.transformer.resblocks[0].mlp.c_fc.weight.dtype


class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fc(x)
        return x



class SSRModule(nn.Module):
    def __init__(self, stage_num=[5, 3], ex_stage_num=0, d=512,
                 class_range=101, lambda_index=1., lambda_delta=1.):
        super(SSRModule, self).__init__()

        self.init_stage_num = stage_num.copy()
        self.stage_num = stage_num
        self.stage_num[0] += ex_stage_num
        
        self.lambda_index = lambda_index
        self.lambda_delta = lambda_delta
        self.class_range = class_range
        self.d = d

        self.stream1_stage2 = Adapter(self.d, 4)
        self.funsion_block_stream1_stage_2_prediction_block = nn.Linear(d, self.stage_num[1])
        self.funsion_block_stream1_stage_1_prediction_block = nn.Linear(d, self.stage_num[0])
    
        self.stream2_stage2 = Adapter(self.d, 4)
        self.funsion_block_stream2_stage_2_prediction_block = nn.Linear(d, self.stage_num[1])
        self.funsion_block_stream2_stage_1_prediction_block = nn.Linear(d, self.stage_num[0])

        self.stage2_FC_after_PB = nn.Sequential(
            nn.Linear(self.stage_num[1], 2 * self.stage_num[1]),
            nn.ReLU()
        )
        self.stage2_prob = nn.Sequential(
            nn.Linear(2 * self.stage_num[1], self.stage_num[1]),
            nn.ReLU()
        )
        self.stage2_index_offsets = nn.Sequential(
            nn.Linear(2 * self.stage_num[1], self.stage_num[1]),
            nn.Tanh()
        )
        self.stage2_delta_k = nn.Sequential(
            nn.Linear(2 * self.stage_num[1], 1),
            nn.Tanh()
        )
        self.stage1_FC_after_PB = nn.Sequential(
            nn.Linear(self.stage_num[0], 2 * self.stage_num[0]),
            nn.ReLU()
        )
        self.stage1_prob = nn.Sequential(
            nn.Linear(2 * self.stage_num[0], self.stage_num[0]),
            nn.ReLU()
        )
        self.stage1_index_offsets = nn.Sequential(
            nn.Linear(2 * self.stage_num[0], self.stage_num[0]),
            nn.Tanh()
        )
        self.stage1_delta_k = nn.Sequential(
            nn.Linear(2 * self.stage_num[0], self.stage_num[0]),
            nn.Tanh()
        )
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0.0)
    
    def forward(self, logits, reasoning_logits, show_importance=False):

        prob_stage_1 = F.softmax(logits, dim=1)
        # reasoning_logits = F.softmax(reasoning_logits, dim=1)
        
        logits = torch.cat((logits, reasoning_logits), dim=1)
        
        embedding_stage1_after_PB = self.stage1_FC_after_PB(logits)
        stage1_delta_k = self.stage1_delta_k(embedding_stage1_after_PB)

        stage1_regress_a = prob_stage_1[:, 0] * 0

        for index in range(self.init_stage_num[0]):
            width = (bin_list_a[index] / (1 + self.lambda_delta * stage1_delta_k[:, index]))
            stage1_regress_a = stage1_regress_a + prob_stage_1[:, index] * width
        stage1_regress_a = torch.unsqueeze(stage1_regress_a, 1)


        regress_age_a = stage1_regress_a
        regress_age_a = regress_age_a.squeeze(1)

        regress_age = regress_age_a
        
        if show_importance:
            # zero‐out any old grads
            if logits.grad is not None: logits.grad.zero_()
            if reasoning_logits.grad is not None: reasoning_logits.grad.zero_()

            # backward to get ∂regress_age / ∂inputs
            # sum over batch so we get a scalar for autograd
            regress_age.sum().backward(retain_graph=True)

            # gradients are now in .grad
            saliency_logits = logits.grad.abs()        # [batch, class_range]
            saliency_reason = reasoning_logits.grad.abs()  # [batch, class_range]

            # you can normalize per‐sample if you like:
            saliency_logits = saliency_logits / (saliency_logits.sum(dim=1, keepdim=True) + 1e-8)
            saliency_reason = saliency_reason / (saliency_reason.sum(dim=1, keepdim=True) + 1e-8)

            return regress_age, saliency_logits, saliency_reason

        return regress_age
    

def load_clip_to_cpu(
    text_encoder_name,
    image_encoder_name,
    root=osp.join(osp.expanduser("~/.cache/clip")),
):
    # text backbone
    if logger is not None:
        print_func = logger.info
    else:
        print_func = print

    print_func("Building CLIP model...")
    text_backbone_name = text_encoder_name
    print_func(f"Text backbone : {text_backbone_name}'s counterpart.")
    url = clip._MODELS[text_backbone_name]
    model_path = clip._download(url, root=root)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    # image backbone
    embed_dim = model.text_projection.shape[1]
    input_resolution = model.visual.input_resolution
    image_backbone_name = image_encoder_name
    print_func(f"Image backbone: {image_backbone_name}")

    if image_backbone_name != text_backbone_name:
        # remove the stochastic back-prop in vgg and alexnet
        MODEL = getattr(image_encoders, image_backbone_name, None)
        if MODEL is None:
            MODEL = getattr(models, image_backbone_name, None)
            logger.warning(f"Try PyTorch Official image model: {image_backbone_name}")
        else:
            logger.info(f"Try Custom image model: {image_backbone_name}")
        if MODEL is None:
            raise ValueError(f"Invalid torchvison model name: {image_backbone_name}")
        model.visual = MODEL(num_classes=embed_dim)
        model.visual.input_resolution = input_resolution
    else:
        print_func(f"CLIP Image encoder: {image_backbone_name}!")

    return model

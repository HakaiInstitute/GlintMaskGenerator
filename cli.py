# Created by: Taylor Denouden
# Organization: Hakai Institute
# Date: 2020-05-30
# Description: Command line interface to the glint-mask-tools.

import fire

from core.glint_mask import rgb_make_and_save_mask, red_edge_make_and_save_mask
from core.specular_mask import make_and_save_mask as specular_make_and_save_mask

if __name__ == '__main__':
    fire.Fire({
        "rgb": rgb_make_and_save_mask,
        "red_edge": red_edge_make_and_save_mask,
        "specular": specular_make_and_save_mask
    })

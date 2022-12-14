# CTN: Deep 3D Vessel Segmentation based on Cross Transformer Network
The coronary microvascular disease poses a great threat to human health. Computer-aided analysis/diagnosis systems help physicians intervene in the disease at early stages, where 3D vessel segmentation is a fundamental step. However, there is a lack of carefully annotated dataset to support algorithm development and evaluation. On the other hand, the commonly-used U-Net structures often yield disconnected and inaccurate segmentation results, especially for small vessel structures. In this paper, motivated by the data scarcity, we first construct two large-scale vessel segmentation datasets consisting of 100 and 500 computed tomography (CT) volumes with pixel-level annotations by experienced radiologists. To enhance the U-Net, we further propose the cross transformer network (CTN) for fine-grained vessel segmentation. In CTN, a transformer module is constructed in parallel to a U-Net to learn long-distance dependencies between different anatomical regions; and these dependencies are communicated to the U-Net at multiple stages to endow it with global awareness. Experimental results on the two in-house datasets indicate that this hybrid model alleviates unexpected disconnections by considering topological information across regions.

## Usage

We mainly use main_c.py (under the project) to train/val/test our models.

The following is one example:
```
python main_c.py --gpu 0 --train --config ./tasks/configs/ccta_vessel.yaml
```
The main parameters are following:
* --train: used to train the model.
* --test: used to test(val) the model.
* --config: the path to the configuration file(*.yaml).
* --resume(optional): the path to the checkpoint pth(resume the model).
* --gpu(default 0): decide to which gpu to select. Format: one or multiple integers(separated by space keys), such as 
: --gpu 0 1 2
* --check_point(optional): the path to save the trained model, we usually specify the parameter in the config file, if 
you specify this parameter here, it will override this parameter in the config file. 

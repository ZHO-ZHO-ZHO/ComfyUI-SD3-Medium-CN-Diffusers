# ComfyUI-SD3-Medium-CN-Diffusers（WIP）

![screenshot-20240616-034804](https://github.com/ZHO-ZHO-ZHO/ComfyUI-SD3-Medium-CN-Diffusers/assets/140084057/a09efa7c-6df0-464d-a7bc-19c3af913a67)


ComfyUI SD3-Medium ControlNet（Diffusers）


已初步能用，但不推荐本地使用（会自动下模型，会有 diffusers 的版本冲突，仅推荐 colab 云上用），原项目 InstantX/SD3-Controlnet- 的代码有问题，自己踩了3个坑，然后还参考了 [kijai](https://github.com/kijai) 的[代码](https://github.com/kijai/ComfyUI-DiffusersSD3Wrapper) 才发现需要 controlnet_start_step 和 controlnet_end_step 这两个参数才能起作用


另外需要手动装这个版本的 diffusers：

git clone -b sd3_control https://github.com/instantX-research/diffusers_sd3_control.git

cd diffusers_sd3_control（原项目代码这里有问题，这是改好的）

pip install -e .

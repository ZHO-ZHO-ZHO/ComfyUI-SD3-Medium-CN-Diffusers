# ComfyUI-SD3-Medium-CN-Diffusers



Tile
![screenshot-20240618-232402](https://github.com/ZHO-ZHO-ZHO/ComfyUI-SD3-Medium-CN-Diffusers/assets/140084057/d1b0df6e-48a6-4849-b115-53a101c7ed9c)


Pose

![screenshot-20240618-234041](https://github.com/ZHO-ZHO-ZHO/ComfyUI-SD3-Medium-CN-Diffusers/assets/140084057/33ed1d5e-0b45-4b23-bde3-d322443233dc)

Canny

![screenshot-20240616-034804](https://github.com/ZHO-ZHO-ZHO/ComfyUI-SD3-Medium-CN-Diffusers/assets/140084057/a09efa7c-6df0-464d-a7bc-19c3af913a67)


## ComfyUI SD3-Medium ControlNet（Diffusers）


20240619

- 原项目把代码重构了，我已根据新代码更新了项目，目前提供 3 种 ControlNet 可用：Pose, Canny 和 Tile

- 需要注意的是目前这版都是从 huggingface 自动下载模型，不太适合本地使用，仅用于尝鲜的云端用户使用

- 显存要求：22GB


20240615 

- 已初步能用，但不推荐本地使用（会自动下模型，会有 diffusers 的版本冲突，仅推荐 colab 云上用），原项目 InstantX/SD3-Controlnet- 的代码有问题，自己踩了3个坑，然后还参考了 [kijai](https://github.com/kijai) 的[代码](https://github.com/kijai/ComfyUI-DiffusersSD3Wrapper) 才发现需要 controlnet_start_step 和 controlnet_end_step 这两个参数才能起作用


- 另外需要手动装这个版本的 diffusers：

   `git clone -b sd3_control https://github.com/instantX-research/diffusers_sd3_control.git`

   `cd diffusers_sd3_control`（原项目代码这里有问题，这是改好的）

   `pip install -e .`


## Credits

https://huggingface.co/InstantX

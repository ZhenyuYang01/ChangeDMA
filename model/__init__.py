from model.encoder import *
from model.ChangeDMA import ChangeDMA

def build_model(img_size=256, num_classes=1, scale="small"):
    if scale == "tiny":
        model = ChangeDMA(img_size=img_size, num_classes=num_classes, dma_depths=[2, 2, 6, 2],
                          dims=[40, 80, 160, 320], mem_dim=80, mem_group=1024)
    elif scale == "small":
        model = ChangeDMA(img_size=img_size, num_classes=num_classes, dma_depths=[2, 2, 6, 2],
                          dims=[48, 96, 192, 384], mem_dim=96, mem_group=196)
    elif scale == "base":
        model = ChangeDMA(img_size=img_size, num_classes=num_classes, dma_depths=[2, 2, 8, 2],
                          dims=[80, 160, 320, 640], mem_dim=160, mem_group=196)
    else:
        raise ValueError(f"Scale {scale} not supported.")
    return model

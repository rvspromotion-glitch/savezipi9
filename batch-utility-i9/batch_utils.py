import logging
import torch


class BatchToList:
    """
    Converts a batched image tensor to a list of individual images.
    Useful for processing each image separately through downstream nodes.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "convert"
    OUTPUT_IS_LIST = (True,)  # Output as list
    CATEGORY = "image/batch"

    def convert(self, images):
        """Convert batched tensor to list of individual images."""
        logger = logging.getLogger("BatchToList")

        batch_size = images.shape[0]
        logger.info(f"Converting batch tensor {images.shape} to list of {batch_size} images")

        # Split along batch dimension
        image_list = []
        for i in range(batch_size):
            image_list.append(images[i:i+1])  # Keep batch dim as [1, H, W, C]

        logger.info(f"✓ Created list of {len(image_list)} images, each shape: {image_list[0].shape}")

        return (image_list,)


class ListToBatch:
    """
    Converts a list of images back to a batched tensor.
    Inverse operation of BatchToList.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "convert"
    INPUT_IS_LIST = (True,)  # Accept list input
    OUTPUT_IS_LIST = (False,)  # Output single tensor
    CATEGORY = "image/batch"

    def convert(self, images):
        """Convert list of images to batched tensor."""
        logger = logging.getLogger("ListToBatch")

        batch_size = len(images)
        logger.info(f"Converting list of {batch_size} images to batched tensor")

        # Concatenate along batch dimension
        batched = torch.cat(images, dim=0)

        logger.info(f"✓ Created batched tensor: {batched.shape}")

        return (batched,)


NODE_CLASS_MAPPINGS = {
    "BatchToList": BatchToList,
    "ListToBatch": ListToBatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BatchToList": "Batch → List",
    "ListToBatch": "List → Batch",
}

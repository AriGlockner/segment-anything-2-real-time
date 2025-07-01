
# if using Apple MPS, fall back to CPU for unsupported ops
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

from sam2.build_sam import build_sam2 #, build_sam2_camera_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor



# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

np.random.seed(3)

# 
sam2_checkpoint = "/home/erie_lab/Repos/segment-anything-2-real-time/checkpoints/sam2.1_hiera_tiny.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_t"

sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)

predictor = SAM2ImagePredictor(sam2_model)
'''predictor = build_sam2_camera_predictor(
    model_cfg, sam2_checkpoint, device=device, vos_optimized=True
)'''


def show_mask(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))    

def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, HTMLResponse
import io
import base64

app = FastAPI()
# TODO: keep using the camera predictor for the first frame, but reset the first frame every xxx often
# Using predictor.track could be super fast

@app.post("/predict")
async def predict(file: UploadFile = File(...),
                  point_x: int = Form(...),
                  point_y: int = Form(...)):
    contents = await file.read()

    # Load the image
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    frame_np = np.array(image).astype(np.uint8)

    # Save the uploaded image as a file
    plt.imsave("demo/images/unity_input.jpg", frame_np)

    # Load the input frame into the predictor
    predictor.set_image(frame_np)

    # Create the input point and label
    input_point = np.array([[point_x, point_y]], dtype=np.float32)
    input_label = np.array([1], dtype=np.int32)

    # Generate masks using the predictor
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )
    sorted_ind = np.argsort(scores)[::-1]
    masks = masks[sorted_ind]
    scores = scores[sorted_ind]
    logits = logits[sorted_ind]
  
    # Save the masks and scores
    best_score = 0.0
    best_mask_img = None

    # Generate and save the best mask image
    for i, (mask, score) in enumerate(zip(masks, scores)):
        if score > best_score:
            # Update the best score and mask
            best_score = score

            # Convert mask to a binary image
            mask_img = (mask > 0.0).astype(np.uint8) * 255
            mask_img_3ch = np.repeat(mask_img[:, :, np.newaxis], 3, axis=2)
            best_mask_img = mask_img_3ch
    
    # Save the best mask image
    if best_mask_img is not None:
        plt.imsave("demo/images/unity_best_mask.jpg", best_mask_img)

    # Save an image with the best mask overlay
    plt.figure(figsize=(10, 10))
    plt.imshow(frame_np)
    show_mask(masks[0], plt.gca(), borders=True)  # Show the first mask
    show_points(input_point, input_label, plt.gca())  # Show the input point
    plt.axis('off')
    plt.savefig("demo/images/unity_mask_overlay.jpg", bbox_inches='tight', pad_inches=0)

    # Convert the mask logits to a byte string for JSON response
    mask_array = np.array(Image.open("demo/images/unity_best_mask.jpg").convert("L"))
    # print(mask_array.tolist())
    # print(mask_array.tolist())


    # show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0]),
    #    "generated_masks": (out_mask_logits > 0.0).cpu().numpy().tolist()
    # "generated_masks": (out_mask_logits > 0.0).cpu().numpy().tolist()
    '''return JSONResponse(content={
        "message": "Prediction completed successfully!",
        "coordinates": {"x": point_x, "y": point_y},
        "mask_array": mask_array.tolist()
    })'''
    # return JSONResponse(content={"message": "Prediction completed successfully!"})
    return JSONResponse(mask_array.tolist())
    

# JSON file can be very slow. Maybe ByteString

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
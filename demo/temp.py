import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

from sam2.build_sam import build_sam2, build_sam2_camera_predictor






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
model_cfg = "configs/sam2.1/sam2.1_hiera_t_512"

sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)

predictor = build_sam2_camera_predictor(
    model_cfg, sam2_checkpoint, device=device, vos_optimized=True
)


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

@app.post("/initialization")
def camera_initialization():
    pass

@app.post("/camera_predictor")
async def camera_predictor(file: UploadFile = File(...)):
    contents = await file.read()
    frame = Image.open(io.BytesIO(contents)).convert("RGB")

    out_obj_ids, out_mask_logits = predictor.track(frame)

    pass


# TODO: keep using the camera predictor for the first frame, but reset the first frame every xxx often

# Using predictor.track could be super fast



@app.post("/predict")
async def predict(file: UploadFile = File(...),
                  point_x: int = Form(...),
                  point_y: int = Form(...)):
    contents = await file.read()

    frame = Image.open(io.BytesIO(contents)).convert("RGB")
    frame_np = np.array(frame).astype(np.uint8)

    predictor.load_first_frame(frame)

    points = np.array([[point_x, point_y]], dtype=np.float32)
    labels = np.array([1], dtype=np.int32)

    _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(
        frame_idx=0,
        obj_id=1,
        points=points,
        labels=labels,
    )

    out_mask_logits: torch.Tensor = out_mask_logits[0].permute(1, 2, 0)  # Get the first mask logits

    plt.imsave("images/unity_input.jpg", frame_np)
    mask_img = (out_mask_logits > 0.0).cpu().numpy().astype(np.uint8) * 255
    mask_img_3ch = np.repeat(mask_img, 3, axis=2)
    plt.imsave("images/unity_mask.jpg", mask_img_3ch)


    # show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])
    return JSONResponse(content={
        "message": "Prediction completed successfully!",
        "coordinates": {"x": point_x, "y": point_y},
        "generated_masks": (out_mask_logits > 0.0).cpu().numpy().tolist()
    })

    '''
    # Load the image
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image_np = np.array(image).astype(np.uint8)

    predictor.set_image(image)

    image_point = np.array([[point_x, point_y]])
    input_label = np.array([1])

    # Generate masks using SAM2
    masks, scores, logits = predictor.predict(
        point_coords=image_point,
        point_labels=input_label,
        multimask_output=True,
    )

    # Get the best mask (highest score)
    best_mask_idx = np.argmax(scores)
    best_mask = masks[best_mask_idx]
    best_score = scores[best_mask_idx]

    print(f"Generated {len(masks)} masks, best score: {best_score:.3f}")

    # TODO: comment out this section
    print(f"Coordinates received: ({point_x}, {point_y})")

    # Create the images directory if it doesn't exist
    os.makedirs("images", exist_ok=True)
    
    # Save the original image as unity_input.jpg
    image.save("images/unity_input.jpg", "JPEG", quality=95)


    # Create matplotlib figure with the point and mask
    plt.figure(figsize=(10, 10))
    plt.imshow(image_np)
    
    # Show the best mask
    show_mask(best_mask, plt.gca(), borders=True)
    
    # Show the input point
    show_points(image_point, input_label, plt.gca())
    
    plt.title(f"SAM2 Prediction - Point: ({point_x}, {point_y}), Score: {best_score:.3f}")
    plt.axis('off')
    
    # Save the result image with mask
    plt.savefig("images/unity_output_with_mask.jpg", bbox_inches='tight', dpi=150)
    plt.close()
    
    # Also save just the mask as a binary image
    mask_image = Image.fromarray((best_mask * 255).astype(np.uint8), mode='L')
    mask_image.save("images/unity_mask.jpg")
    # TODO: End section to comment out
    
    # "mask_score": float(best_score),
    # "num_masks_generated": len(masks),
        
    return JSONResponse(content={
        "message": "SAM2 prediction completed successfully!",
        "coordinates": {"x": point_x, "y": point_y},
        "saved_files": [
            "images/unity_input.jpg", 
            "images/unity_output_with_mask.jpg",
            "images/unity_mask.jpg"
        ]
    })
    '''

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
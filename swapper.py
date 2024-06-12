import copy
import argparse
import pathlib

import cv2
import numpy as np
from PIL import Image
import onnxruntime
import insightface
from insightface.model_zoo import RetinaFace, Landmark, Attribute, ArcFaceONNX
from insightface.model_zoo.inswapper import INSwapper

type FaceSwapper = RetinaFace | Landmark | Attribute | ArcFaceONNX | INSwapper

ROOT_DIR = pathlib.Path(__file__).parent.resolve()
CHECKPOINTS_DIR = ROOT_DIR / "checkpoints"
MODELS_DIR = CHECKPOINTS_DIR / "models"
CODEFORMER_DIR = ROOT_DIR / "CodeFormer"


def get_face_analyzer(
    model_name: str,
    providers: list[str],
    det_size: tuple[int, int] = (320, 320),
):
    face_analyzer = insightface.app.FaceAnalysis(
        name=model_name, root=CHECKPOINTS_DIR, providers=providers
    )
    face_analyzer.prepare(ctx_id=0, det_size=det_size)
    return face_analyzer


def get_face_swap_model(model_path: str | pathlib.Path) -> FaceSwapper | None:
    return insightface.model_zoo.get_model(str(model_path))


def get_one_face(face_analyzer: insightface.app.FaceAnalysis, frame: np.ndarray):
    face = face_analyzer.get(frame)
    try:
        return min(face, key=lambda x: x.bbox[0])
    except ValueError:
        return None


def get_many_faces(face_analyzer: insightface.app.FaceAnalysis, frame: np.ndarray):
    """Get faces from left to right by order"""
    try:
        face = face_analyzer.get(frame)
        return sorted(face, key=lambda x: x.bbox[0])
    except IndexError:
        return []


def swap_face(
    face_swapper: FaceSwapper,
    source_face,
    target_face,
    temp_frame: np.ndarray,
):
    return face_swapper.get(temp_frame, target_face, source_face, paste_back=True)


def swap_faces(
    source_img: list[Image.Image],
    target_img: Image.Image,
    face_analyzer: insightface.app.FaceAnalysis,
    face_swapper: FaceSwapper,
):
    # read target image
    target_img = cv2.cvtColor(np.array(target_img), cv2.COLOR_RGB2BGR)

    # detect faces that will be replaced in the target image
    target_faces = get_many_faces(face_analyzer, target_img)
    num_target_faces = len(target_faces)
    num_source_images = len(source_img)

    if num_target_faces == 0:
        raise Exception("No target faces found!")

    temp_frame = copy.deepcopy(target_img)

    if num_source_images == num_target_faces:
        print("Replacing faces in target image from the left to the right by order")
        for i in range(num_source_images):
            source_faces = get_many_faces(
                face_analyzer,
                cv2.cvtColor(np.array(source_img[i]), cv2.COLOR_RGB2BGR),
            )

            if source_faces is None:
                raise Exception("No source faces found!")

            temp_frame = swap_face(
                face_swapper,
                source_faces[i],
                target_faces[i],
                temp_frame,
            )

    return Image.fromarray(cv2.cvtColor(temp_frame, cv2.COLOR_BGR2RGB))


def enhance_image(
    image: Image.Image,
    background_enhance: bool = False,
    face_upsample: bool = False,
    upscale: int = 1,
    codeformer_fidelity: float = 0.5,
):
    from restoration import (
        verify_checkpoints,
        set_realesrgan,
        torch,
        ARCH_REGISTRY,
        face_restoration,
    )

    # make sure the ckpts downloaded successfully
    verify_checkpoints()

    # https://huggingface.co/spaces/sczhou/CodeFormer
    upsampler = set_realesrgan()
    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    codeformer_net = ARCH_REGISTRY.get("CodeFormer")(
        dim_embd=512,
        codebook_size=1024,
        n_head=8,
        n_layers=9,
        connect_list=["32", "64", "128", "256"],
    ).to(device)

    ckpt_path = CODEFORMER_DIR / "CodeFormer/weights/CodeFormer/codeformer.pth"
    # ckpt_path = "CodeFormer/CodeFormer/weights/CodeFormer/codeformer.pth"
    checkpoint = torch.load(ckpt_path)["params_ema"]
    codeformer_net.load_state_dict(checkpoint)
    codeformer_net.eval()

    result_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    result_image = face_restoration(
        result_image,
        background_enhance,
        face_upsample,
        upscale,
        codeformer_fidelity,
        upsampler,
        codeformer_net,
        device,
    )
    return Image.fromarray(result_image)


def parse_args():
    parser = argparse.ArgumentParser(description="Face swap.")
    parser.add_argument(
        "--source_img",
        type=str,
        required=True,
        help="The path of source image, it can be multiple images, dir;dir2;dir3.",
    )
    parser.add_argument(
        "--target_img", type=str, required=True, help="The path of target image."
    )
    parser.add_argument(
        "--output_img",
        type=str,
        required=False,
        default="result.png",
        help="The path and filename of output image.",
    )
    parser.add_argument(
        "--face_restore", action="store_true", help="The flag for face restoration."
    )
    parser.add_argument(
        "--background_enhance",
        action="store_true",
        help="The flag for background enhancement.",
    )
    parser.add_argument(
        "--face_upsample", action="store_true", help="The flag for face upsample."
    )
    parser.add_argument(
        "--upscale", type=int, default=1, help="The upscale value, up to 4."
    )
    parser.add_argument(
        "--codeformer_fidelity",
        type=float,
        default=0.5,
        help="The codeformer fidelity.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    face_analyzer = get_face_analyzer(
        model_name="buffalo_l",
        providers=["CPUExecutionProvider"],
        # providers=onnxruntime.get_available_providers(),
    )

    face_swapper = get_face_swap_model(model_path=MODELS_DIR / "inswapper_128.onnx")

    source_img = [Image.open(img_path) for img_path in args.source_img.split(";")]
    target_img = Image.open(args.target_img)

    result_img = swap_faces(
        source_img=source_img,
        target_img=target_img,
        face_analyzer=face_analyzer,
        face_swapper=face_swapper,
    )

    if args.face_restore:
        result_img = enhance_image(
            image=result_img,
            background_enhance=args.background_enhance,
            face_upsample=args.face_upsample,
            upscale=args.upscale,
            codeformer_fidelity=args.codeformer_fidelity,
        )

    result_img.save(args.output_img)
    print(f"Result saved successfully: {args.output_img}")


if __name__ == "__main__":
    main()

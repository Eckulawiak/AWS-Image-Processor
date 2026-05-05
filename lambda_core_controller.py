import io
import json
import logging
import os
import zipfile
from datetime import datetime, timezone
from decimal import Decimal

import boto3
from PIL import Image

# Logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# AWS clients
REGION = os.environ.get("AWS_REGION", "us-east-1")

s3 = boto3.client("s3", region_name=REGION)
rekognition = boto3.client("rekognition", region_name=REGION)
dynamodb = boto3.resource("dynamodb", region_name=REGION)

# Config
PROCESSED_BUCKET = os.environ["PROCESSED_BUCKET"]
DYNAMO_TABLE = os.environ["DYNAMO_TABLE"]
REKOGNITION_MODE = os.environ.get("REKOGNITION_MODE", "all").lower()
TARGET_SIZE = (
    int(os.environ.get("TARGET_WIDTH", 640)),
    int(os.environ.get("TARGET_HEIGHT", 640)),
)
MIN_CONFIDENCE = float(os.environ.get("MIN_CONFIDENCE", 70.0))
MAX_LABELS = int(os.environ.get("MAX_LABELS", 20))

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tiff"}

table = dynamodb.Table(DYNAMO_TABLE)


# Fetch / delete from S3
def fetch_s3_object(bucket: str, key: str) -> bytes:
    logger.info("Fetching s3://%s/%s", bucket, key)
    return s3.get_object(Bucket=bucket, Key=key)["Body"].read()


def delete_raw_object(bucket: str, key: str) -> None:
    s3.delete_object(Bucket=bucket, Key=key)
    logger.info("Deleted raw object s3://%s/%s", bucket, key)


# Unpack: zip
def is_image_key(key: str) -> bool:
    return any(key.lower().endswith(ext) for ext in SUPPORTED_EXTENSIONS)


def unpack(key: str, raw_bytes: bytes) -> list[tuple[str, bytes]]:
    """
    Returns a list of (relative_filename, image_bytes) tuples.
    Handles both .zip archives and single image uploads.
    """
    if key.lower().endswith(".zip"):
        return unpack_zip(key, raw_bytes)
    elif is_image_key(key):
        return [(os.path.basename(key), raw_bytes)]
    else:
        raise ValueError(f"Unsupported file type for key: {key}")


def unpack_zip(zip_key: str, zip_bytes: bytes) -> list[tuple[str, bytes]]:
    """Extract all supported images from a zip archive."""
    images = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for member in zf.infolist():
            # Skip directories and hidden/macOS metadata files
            if member.is_dir() or os.path.basename(member.filename).startswith("._"):
                continue
            if not is_image_key(member.filename):
                logger.info("Skipping non-image zip member: %s", member.filename)
                continue
            logger.info("Extracting from zip: %s (%d bytes)", member.filename, member.file_size)
            image_bytes = zf.read(member.filename)
            images.append((member.filename, image_bytes))

    if not images:
        raise ValueError(f"No supported images found in zip: {zip_key}")

    logger.info("Extracted %d image(s) from zip", len(images))
    return images


# Resize to 640x640 (letterbox)
def resize_image(image_bytes: bytes) -> bytes:
    """
    Resize to TARGET_SIZE using letterboxing (black bars) to preserve
    aspect ratio. Returns JPEG bytes.
    """
    with Image.open(io.BytesIO(image_bytes)) as img:
        img = img.convert("RGB")
        img.thumbnail(TARGET_SIZE, Image.LANCZOS)

        canvas = Image.new("RGB", TARGET_SIZE, (0, 0, 0))
        offset = (
            (TARGET_SIZE[0] - img.width) // 2,
            (TARGET_SIZE[1] - img.height) // 2,
        )
        canvas.paste(img, offset)

        buf = io.BytesIO()
        canvas.save(buf, format="JPEG", quality=95)
        return buf.getvalue()


# Upload resized image to S3 Processed
def upload_processed(image_bytes: bytes, source_key: str, filename: str) -> str:
    """
    Upload resized image to the Processed bucket.
    Key structure: processed/<source_key_stem>/<filename>.jpg
    Returns the full S3 key.
    """
    stem = source_key.rsplit(".", 1)[0]
    base = os.path.splitext(os.path.basename(filename))[0]
    processed_key = f"processed/{stem}/{base}.jpg"

    s3.put_object(
        Bucket=PROCESSED_BUCKET,
        Key=processed_key,
        Body=image_bytes,
        ContentType="image/jpeg",
    )
    logger.info("Uploaded resized image -> s3://%s/%s", PROCESSED_BUCKET, processed_key)
    return processed_key


# Upload YOLO label files to S3 Processed
def upload_label_file(
    annotations: list[str],
    class_names: list[str],
    source_key: str,
    filename: str,
) -> str | None:
    """
    Write a YOLO-format .txt label file and a companion _classes.txt to the
    Processed bucket alongside the resized image.

    YOLO format per line: <class_id> <x_center> <y_center> <width> <height>
    All values are normalized to [0, 1] relative to image dimensions.

    Returns the label S3 key, or None if Rekognition found no bounding boxes.
    """
    if not annotations:
        logger.info("No bounding boxes detected for %s — skipping label upload", filename)
        return None

    stem = source_key.rsplit(".", 1)[0]
    base = os.path.splitext(os.path.basename(filename))[0]
    label_key = f"processed/{stem}/{base}.txt"
    classes_key = f"processed/{stem}/{base}_classes.txt"

    s3.put_object(
        Bucket=PROCESSED_BUCKET,
        Key=label_key,
        Body="\n".join(annotations).encode(),
        ContentType="text/plain",
    )
    logger.info("Uploaded label file -> s3://%s/%s", PROCESSED_BUCKET, label_key)

    # Companion file maps integer class IDs to human-readable names.
    # Line N (0-indexed) is the name for class ID N.
    s3.put_object(
        Bucket=PROCESSED_BUCKET,
        Key=classes_key,
        Body="\n".join(class_names).encode(),
        ContentType="text/plain",
    )
    logger.info("Uploaded classes file -> s3://%s/%s", PROCESSED_BUCKET, classes_key)

    return label_key


# Rekognition
def detect_labels(image_bytes: bytes) -> tuple[list[dict], list[str], list[str]]:
    """
    Call Rekognition detect_labels and extract both metadata and bounding boxes.

    Returns:
        labels_info  — list of dicts written to DynamoDB
        annotations  — YOLO-format strings: "<class_id> <cx> <cy> <w> <h>"
        class_names  — ordered list where index == class_id used in annotations
    """
    response = rekognition.detect_labels(
        Image={"Bytes": image_bytes},
        MaxLabels=MAX_LABELS,
        MinConfidence=MIN_CONFIDENCE,
    )

    labels_info: list[dict] = []
    class_names: list[str] = []
    annotations: list[str] = []

    for lbl in response.get("Labels", []):
        labels_info.append({
            "name": lbl["Name"],
            "confidence": round(lbl["Confidence"], 2),
            "categories": [c["Name"] for c in lbl.get("Categories", [])],
        })

        # Rekognition returns per-instance bounding boxes only for labels that
        # represent discrete physical objects (not scene-level concepts).
        for instance in lbl.get("Instances", []):
            bb = instance.get("BoundingBox")
            if not bb:
                continue

            name = lbl["Name"]
            if name not in class_names:
                class_names.append(name)
            class_id = class_names.index(name)

            # Rekognition uses (Left, Top, Width, Height) all normalized 0-1.
            # YOLO expects (x_center, y_center, width, height) normalized 0-1.
            x_center = bb["Left"] + bb["Width"] / 2
            y_center = bb["Top"] + bb["Height"] / 2
            annotations.append(
                f"{class_id} {x_center:.6f} {y_center:.6f} {bb['Width']:.6f} {bb['Height']:.6f}"
            )

    logger.info(
        "Rekognition labels: %d detected, %d bounding boxes across %d classes",
        len(labels_info), len(annotations), len(class_names),
    )
    return labels_info, annotations, class_names


def detect_moderation(image_bytes: bytes) -> tuple[list[dict], bool]:
    response = rekognition.detect_moderation_labels(
        Image={"Bytes": image_bytes},
        MinConfidence=MIN_CONFIDENCE,
    )
    mod_labels = [
        {
            "name": lbl["Name"],
            "parent": lbl.get("ParentName", ""),
            "confidence": round(lbl["Confidence"], 2),
        }
        for lbl in response.get("ModerationLabels", [])
    ]
    return mod_labels, len(mod_labels) == 0


def run_rekognition(image_bytes: bytes) -> tuple[dict, list[str], list[str]]:
    """
    Returns (rek_result_for_dynamo, annotations, class_names).
    annotations and class_names are empty when REKOGNITION_MODE excludes labels.
    """
    result = {"labels": [], "moderation_labels": [], "is_safe": True}
    annotations: list[str] = []
    class_names: list[str] = []

    if REKOGNITION_MODE in ("labels", "all"):
        result["labels"], annotations, class_names = detect_labels(image_bytes)

    if REKOGNITION_MODE in ("moderation", "all"):
        result["moderation_labels"], result["is_safe"] = detect_moderation(image_bytes)

    return result, annotations, class_names


# DynamoDB write
def _floats_to_decimal(obj):
    """DynamoDB SDK requires Decimal instead of float."""
    if isinstance(obj, float):
        return Decimal(str(obj))
    if isinstance(obj, dict):
        return {k: _floats_to_decimal(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_floats_to_decimal(i) for i in obj]
    return obj


def write_dynamo(
    job_id: str,
    image_key: str,
    source_bucket: str,
    rek_results: dict,
    label_key: str | None = None,
    status: str = "PROCESSED",
    error: str = None,
):
    item = {
        "job_id": job_id,           # PK — source S3 key (zip or image)
        "image_key": image_key,     # SK — resized image key in Processed bucket
        "status": status,
        "source_bucket": source_bucket,
        "processed_bucket": PROCESSED_BUCKET,
        "processed_at": datetime.now(timezone.utc).isoformat(),
        **_floats_to_decimal(rek_results),
    }
    if label_key:
        item["label_key"] = label_key
    if error:
        item["error"] = error

    table.put_item(Item=item)
    logger.info("DynamoDB record written: job_id=%s image_key=%s", job_id, image_key)


# Lambda handler
def lambda_handler(event, context):
    """
    Accepts EventBridge S3 ObjectCreated events or direct S3-style payloads.

    EventBridge envelope:
      event["detail"]["bucket"]["name"]
      event["detail"]["object"]["key"]

    Direct / test invocation:
      event["bucket"] + event["key"]
    """
    logger.info("Event: %s", json.dumps(event, default=str))

    # Resolve bucket + key
    if "detail" in event:
        # EventBridge S3 notification
        bucket = event["detail"]["bucket"]["name"]
        key = event["detail"]["object"]["key"]
    elif "bucket" in event and "key" in event:
        bucket = event["bucket"]
        key = event["key"]
    elif "Records" in event:
        # Legacy S3 direct trigger (fallback)
        record = event["Records"][0]["s3"]
        bucket = record["bucket"]["name"]
        key = record["object"]["key"]
    else:
        raise ValueError("Unrecognised event shape — expected EventBridge, S3, or direct invocation.")

    logger.info("Processing job: s3://%s/%s", bucket, key)

    # Fetch
    raw_bytes = fetch_s3_object(bucket, key)

    # Unpack
    images = unpack(key, raw_bytes)  # [(filename, bytes), ...]

    # Process each image
    summary = {"job_id": key, "total": len(images), "processed": 0, "failed": 0, "results": []}

    for filename, img_bytes in images:
        image_key = "pending"
        try:
            # Resize
            resized_bytes = resize_image(img_bytes)

            # Upload to Processed bucket
            image_key = upload_processed(resized_bytes, key, filename)

            # Rekognition
            rek_results, annotations, class_names = run_rekognition(resized_bytes)

            # Upload YOLO label file (skipped if no bounding boxes detected)
            label_key = upload_label_file(annotations, class_names, key, filename)

            # DynamoDB
            write_dynamo(
                job_id=key,
                image_key=image_key,
                source_bucket=bucket,
                rek_results=rek_results,
                label_key=label_key,
                status="PROCESSED",
            )

            summary["processed"] += 1
            summary["results"].append({
                "filename": filename,
                "image_key": image_key,
                "label_key": label_key,
                "status": "PROCESSED",
            })

        except Exception as exc:
            logger.exception("Failed to process image: %s", filename)
            summary["failed"] += 1
            summary["results"].append({"filename": filename, "status": "FAILED", "error": str(exc)})

            # Still write a failed record to DynamoDB for observability
            try:
                write_dynamo(
                    job_id=key,
                    image_key=image_key,
                    source_bucket=bucket,
                    rek_results={},
                    status="FAILED",
                    error=str(exc),
                )
            except Exception:
                logger.exception("Failed to write FAILED record to DynamoDB for: %s", filename)

    # Delete the raw source object from the input bucket now that all images
    # have been processed (or attempted). Runs regardless of per-image failures
    # so that a partially-failed zip does not get reprocessed automatically.
    try:
        delete_raw_object(bucket, key)
    except Exception:
        logger.exception("Failed to delete raw object s3://%s/%s — manual cleanup required", bucket, key)

    logger.info("Job complete: %s", json.dumps(summary, default=str))
    return {"statusCode": 200, **summary}

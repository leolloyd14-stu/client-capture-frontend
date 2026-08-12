from contextlib import asynccontextmanager
from datetime import datetime
from difflib import get_close_matches
import re

import cv2
import numpy as np

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from paddleocr import PaddleOCR


# =========================================================
# GLOBAL OCR ENGINE
# =========================================================

ocr_engine = None


# =========================================================
# FORM CONFIGURATION
# =========================================================

ROW_FIELD_MAP = {
    1: "ownerName",
    2: None,
    3: "mobileNumber",
    4: "address",
    5: "nationality",
    6: "idNumber",
    7: "idType",
    8: "issuingDate",
    9: "expiringDate",
    10: "union",
}

EXPECTED_ROW_COUNT = 10

CONFIDENCE_REVIEW_THRESHOLD = 0.85

MAX_IMAGE_SIZE_BYTES = (
    10 * 1024 * 1024
)


# =========================================================
# APPLICATION LIFESPAN
# =========================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global ocr_engine

    print("Loading PaddleOCR...")

    ocr_engine = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )

    print("PaddleOCR loaded successfully.")

    yield

    print("OCR API shutting down...")


# =========================================================
# FASTAPI
# =========================================================

app = FastAPI(
    title="Client Capture OCR API",
    version="1.2.0",
    lifespan=lifespan,
)


# =========================================================
# CORS
#
# Local development now.
# Add GitHub Pages URL later before deployment.
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "https://leolloyd14-stu.github.io/client-capture-frontend/", 
    ],
    allow_credentials=True,
    allow_methods=[
        "GET",
        "POST",
        "OPTIONS",
    ],
    allow_headers=["*"],
)


# =========================================================
# HEALTH ENDPOINTS
# =========================================================

@app.get("/")
async def root():
    return {
        "status": "ok",
        "service": "client-capture-ocr-api",
        "version": "1.2.0",
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "ocr_ready": ocr_engine is not None,
    }


# =========================================================
# BASIC TEXT NORMALIZATION
# =========================================================

def normalize_text(
    value: str,
) -> str:
    if not value:
        return ""

    value = value.strip()

    return re.sub(
        r"\s+",
        " ",
        value,
    )


# =========================================================
# NUMERIC OCR CORRECTION
# =========================================================

def normalize_numeric_ocr(
    value: str,
) -> str:
    value = normalize_text(
        value
    )

    translation = str.maketrans({
        "O": "0",
        "o": "0",
        "I": "1",
        "l": "1",
        "|": "1",
    })

    return value.translate(
        translation
    )


# =========================================================
# MOBILE NUMBER
# =========================================================

def normalize_mobile_number(
    value: str,
) -> str:
    value = normalize_numeric_ocr(
        value
    )

    return re.sub(
        r"\D",
        "",
        value,
    )


# =========================================================
# ID NUMBER
# =========================================================

def normalize_id_number(
    value: str,
) -> str:
    value = normalize_text(
        value
    )

    return re.sub(
        r"\s+",
        "",
        value,
    )


# =========================================================
# DATE
# =========================================================

def normalize_date(
    value: str,
) -> str:
    raw = normalize_numeric_ocr(
        value
    )

    raw = (
        raw
        .replace("/", "-")
        .replace(".", "-")
        .replace("_", "-")
    )

    raw = re.sub(
        r"\s+",
        "",
        raw,
    )

    formats = [
        "%d-%m-%Y",
        "%Y-%m-%d",
        "%d-%m-%y",
    ]

    for date_format in formats:
        try:
            parsed = datetime.strptime(
                raw,
                date_format,
            )

            return parsed.strftime(
                "%Y-%m-%d"
            )

        except ValueError:
            continue

    return normalize_text(
        value
    )


# =========================================================
# FUZZY MATCHING
# =========================================================

def fuzzy_choice(
    value: str,
    candidates: dict[str, str],
    cutoff: float,
) -> str:
    clean = re.sub(
        r"[^A-Z0-9]",
        "",
        normalize_text(value).upper(),
    )

    if not clean:
        return ""

    if clean in candidates:
        return candidates[clean]

    matches = get_close_matches(
        clean,
        list(candidates.keys()),
        n=1,
        cutoff=cutoff,
    )

    if matches:
        return candidates[
            matches[0]
        ]

    return normalize_text(
        value
    )


# =========================================================
# ID TYPE
# =========================================================

def normalize_id_type(
    value: str,
) -> str:
    candidates = {
        "GHANACARD":
            "Ghana Card",

        "PASSPORT":
            "Passport",

        "DRIVERSLICENCE":
            "Driver's Licence",

        "DRIVERLICENCE":
            "Driver's Licence",

        "DRIYERSLICENCE":
            "Driver's Licence",

        "DRIVERSLICENSE":
            "Driver's Licence",

        "DRIVERLICENSE":
            "Driver's Licence",

        "VOTERID":
            "Voter ID",
    }

    return fuzzy_choice(
        value,
        candidates,
        cutoff=0.68,
    )


# =========================================================
# NATIONALITY
# =========================================================

def normalize_nationality(
    value: str,
) -> str:
    candidates = {
        "GHANAIAN":
            "Ghanaian",

        "NIGERIAN":
            "Nigerian",

        "IVORIAN":
            "Ivorian",

        "TOGOLESE":
            "Togolese",

        "BURKINABE":
            "Burkinabé",

        "BURKINAB":
            "Burkinabé",

        "BURKINBE":
            "Burkinabé",

        "BURKINTBE":
            "Burkinabé",

        "BURKINATBE":
            "Burkinabé",
    }

    return fuzzy_choice(
        value,
        candidates,
        cutoff=0.64,
    )


# =========================================================
# NORMALIZE FIELD
# =========================================================

def normalize_field_value(
    field_name: str,
    value: str,
) -> str:
    if field_name == "mobileNumber":
        return normalize_mobile_number(
            value
        )

    if field_name == "idNumber":
        return normalize_id_number(
            value
        )

    if field_name == "idType":
        return normalize_id_type(
            value
        )

    if field_name == "nationality":
        return normalize_nationality(
            value
        )

    if field_name in {
        "issuingDate",
        "expiringDate",
    }:
        return normalize_date(
            value
        )

    return normalize_text(
        value
    )


# =========================================================
# FIELD VALIDATION
# =========================================================

def validate_field(
    field_name: str,
    value: str,
) -> list[str]:
    warnings = []

    if not value:
        warnings.append(
            "No value was detected."
        )

        return warnings


    if field_name == "mobileNumber":
        digits = re.sub(
            r"\D",
            "",
            value,
        )

        if not 8 <= len(digits) <= 15:
            warnings.append(
                "Mobile number length appears invalid."
            )


    elif field_name in {
        "issuingDate",
        "expiringDate",
    }:
        try:
            datetime.strptime(
                value,
                "%Y-%m-%d",
            )

        except ValueError:
            warnings.append(
                "Date could not be normalized reliably."
            )


    elif field_name == "idType":
        valid_types = {
            "Ghana Card",
            "Passport",
            "Driver's Licence",
            "Voter ID",
        }

        if value not in valid_types:
            warnings.append(
                "ID type was not matched to a configured ID type."
            )


    return warnings


# =========================================================
# OCR BOX HELPERS
# =========================================================

def get_box_center_y(
    box,
) -> float | None:
    if (
        not box or
        len(box) < 4
    ):
        return None

    return (
        float(box[1]) +
        float(box[3])
    ) / 2.0


def get_box_center_x(
    box,
) -> float:
    if (
        not box or
        len(box) < 4
    ):
        return 0.0

    return (
        float(box[0]) +
        float(box[2])
    ) / 2.0


# =========================================================
# HORIZONTAL GRID DETECTION
# =========================================================

def detect_horizontal_grid_lines(
    image: np.ndarray,
) -> list[int]:
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        15,
    )

    image_width = (
        image.shape[1]
    )

    kernel_width = max(
        30,
        image_width // 8,
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (
            kernel_width,
            1,
        ),
    )

    horizontal = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel,
    )

    contours, _ = (
        cv2.findContours(
            horizontal,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
    )

    candidate_y = []

    for contour in contours:
        x, y, width, height = (
            cv2.boundingRect(
                contour
            )
        )

        if (
            width >=
            image_width * 0.35
        ):
            candidate_y.append(
                int(
                    y +
                    height / 2
                )
            )


    candidate_y.sort()


    merged = []

    tolerance = max(
        4,
        image.shape[0] // 150,
    )


    for y in candidate_y:
        if not merged:
            merged.append([y])
            continue


        previous_average = (
            sum(merged[-1]) /
            len(merged[-1])
        )


        if (
            abs(
                y -
                previous_average
            )
            <= tolerance
        ):
            merged[-1].append(
                y
            )
        else:
            merged.append(
                [y]
            )


    return [
        int(
            round(
                sum(group) /
                len(group)
            )
        )
        for group in merged
    ]


# =========================================================
# FORM ROW BOUNDARIES
# =========================================================

def build_row_boundaries(
    image: np.ndarray,
) -> tuple[
    list[tuple[int, int]],
    str,
]:
    image_height = (
        image.shape[0]
    )

    lines = (
        detect_horizontal_grid_lines(
            image
        )
    )


    # Ten rows require eleven boundary lines.
    if len(lines) >= 11:
        best_sequence = None
        best_score = None


        for start in range(
            0,
            len(lines) - 10,
        ):
            sequence = (
                lines[
                    start:
                    start + 11
                ]
            )

            gaps = np.diff(
                sequence
            )


            if np.any(
                gaps <= 0
            ):
                continue


            mean_gap = float(
                np.mean(gaps)
            )


            if mean_gap <= 0:
                continue


            variation = float(
                np.std(gaps) /
                mean_gap
            )


            # Prefer consistent boundaries.
            if (
                best_score is None or
                variation < best_score
            ):
                best_score = (
                    variation
                )

                best_sequence = (
                    sequence
                )


        if best_sequence:
            rows = []

            for index in range(
                EXPECTED_ROW_COUNT
            ):
                rows.append(
                    (
                        int(
                            best_sequence[
                                index
                            ]
                        ),

                        int(
                            best_sequence[
                                index + 1
                            ]
                        ),
                    )
                )

            return (
                rows,
                "detected_grid",
            )


    # =====================================================
    # FALLBACK
    #
    # If grid detection fails, use 10 proportional rows.
    # This preserves field positions and prevents sequential
    # OCR-shifting.
    # =====================================================

    row_height = (
        image_height /
        EXPECTED_ROW_COUNT
    )

    rows = []

    for index in range(
        EXPECTED_ROW_COUNT
    ):
        top = int(
            round(
                index *
                row_height
            )
        )

        bottom = int(
            round(
                (index + 1) *
                row_height
            )
        )

        rows.append(
            (
                top,
                bottom,
            )
        )

    return (
        rows,
        "proportional_fallback",
    )


# =========================================================
# LOCATE PHYSICAL ROW
# =========================================================

def find_row_number(
    center_y: float,
    row_boundaries:
        list[tuple[int, int]],
) -> int | None:
    for index, (
        top,
        bottom,
    ) in enumerate(
        row_boundaries,
        start=1,
    ):
        if (
            top <=
            center_y <
            bottom
        ):
            return index

    return None


# =========================================================
# GROUP OCR ITEMS BY ROW
# =========================================================

def group_items_by_row(
    detected_items:
        list[dict],

    row_boundaries:
        list[tuple[int, int]],

) -> tuple[
    dict[int, list[dict]],
    list[dict],
]:
    grouped = {
        row: []
        for row in range(
            1,
            EXPECTED_ROW_COUNT + 1
        )
    }

    unmapped = []


    for item in detected_items:
        center_y = (
            get_box_center_y(
                item.get("box")
            )
        )


        if center_y is None:
            item[
                "mapping_reason"
            ] = (
                "OCR item has no usable bounding box."
            )

            unmapped.append(
                item
            )

            continue


        row_number = (
            find_row_number(
                center_y,
                row_boundaries,
            )
        )


        if row_number is None:
            item[
                "mapping_reason"
            ] = (
                "OCR item falls outside the detected form rows."
            )

            unmapped.append(
                item
            )

            continue


        item[
            "physical_row"
        ] = row_number


        grouped[
            row_number
        ].append(
            item
        )


    # Left-to-right ordering within each row.
    for items in (
        grouped.values()
    ):
        items.sort(
            key=lambda item:
                get_box_center_x(
                    item.get("box")
                )
        )


    return (
        grouped,
        unmapped,
    )


# =========================================================
# MERGE OCR ITEMS FROM SAME ROW
# =========================================================

def merge_row_items(
    row_items: list[dict],
) -> dict:
    if not row_items:
        return {
            "text": "",
            "confidence": None,
            "boxes": [],
            "source_count": 0,
        }


    texts = []

    for item in row_items:
        text = normalize_text(
            item.get(
                "text",
                "",
            )
        )

        if text:
            texts.append(
                text
            )


    merged_text = (
        " ".join(texts)
    )


    weighted_total = 0.0
    total_weight = 0


    for item in row_items:
        confidence = (
            item.get(
                "confidence"
            )
        )

        text = normalize_text(
            item.get(
                "text",
                "",
            )
        )


        if confidence is None:
            continue


        weight = max(
            len(text),
            1,
        )


        weighted_total += (
            float(confidence) *
            weight
        )


        total_weight += (
            weight
        )


    average_confidence = (
        weighted_total /
        total_weight

        if total_weight
        else None
    )


    return {
        "text":
            merged_text,

        "confidence":
            average_confidence,

        "boxes": [
            item.get(
                "box"
            )
            for item
            in row_items
        ],

        "source_count":
            len(row_items),
    }


# =========================================================
# MAP ROWS TO FORM FIELDS
# =========================================================

def map_rows_to_fields(
    grouped_rows:
        dict[int, list[dict]],
) -> dict:
    fields = {}


    for (
        row_number,
        field_name,
    ) in ROW_FIELD_MAP.items():

        # Row 2 is intentionally ignored.
        if field_name is None:
            continue


        merged = (
            merge_row_items(
                grouped_rows.get(
                    row_number,
                    [],
                )
            )
        )


        raw_value = (
            merged["text"]
        )


        normalized_value = (
            normalize_field_value(
                field_name,
                raw_value,
            )
        )


        confidence = (
            merged[
                "confidence"
            ]
        )


        warnings = (
            validate_field(
                field_name,
                normalized_value,
            )
        )


        if (
            confidence is None
        ):
            if (
                "No value was detected."
                not in warnings
            ):
                warnings.append(
                    "No OCR confidence score is available."
                )

        elif (
            confidence <
            CONFIDENCE_REVIEW_THRESHOLD
        ):
            warnings.append(
                "OCR confidence is below the review threshold."
            )


        fields[
            field_name
        ] = {
            "row":
                row_number,

            "value":
                normalized_value,

            "raw_value":
                raw_value,

            "confidence":
                confidence,

            "requires_review":
                bool(warnings),

            "warnings":
                warnings,

            "source_count":
                merged[
                    "source_count"
                ],

            "boxes":
                merged[
                    "boxes"
                ],
        }


    return fields


# =========================================================
# OCR ENDPOINT
# =========================================================

@app.post("/ocr")
async def extract_text(
    image: UploadFile = File(...)
):
    allowed_types = {
        "image/jpeg",
        "image/png",
        "image/webp",
    }


    if (
        image.content_type
        not in allowed_types
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "Only JPEG, PNG, and WebP images are allowed."
            ),
        )


    contents = (
        await image.read()
    )


    if not contents:
        raise HTTPException(
            status_code=400,
            detail=(
                "The uploaded image is empty."
            ),
        )


    if (
        len(contents) >
        MAX_IMAGE_SIZE_BYTES
    ):
        raise HTTPException(
            status_code=413,
            detail=(
                "The uploaded image exceeds the 10 MB limit."
            ),
        )


    image_array = (
        np.frombuffer(
            contents,
            dtype=np.uint8,
        )
    )


    decoded_image = (
        cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR,
        )
    )


    if decoded_image is None:
        raise HTTPException(
            status_code=400,
            detail=(
                "The uploaded file could not be decoded as an image."
            ),
        )


    if ocr_engine is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "OCR engine is not ready."
            ),
        )


    # =====================================================
    # PADDLEOCR
    # =====================================================

    try:
        results = (
            ocr_engine.predict(
                decoded_image
            )
        )

    except Exception as error:
        print(
            "PaddleOCR error:",
            error,
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "OCR processing failed."
            ),
        )


    detected_items = []


    # =====================================================
    # EXTRACT OCR RESULTS
    # =====================================================

    for result in results:
        data = result.json


        if (
            "res" in data
            and isinstance(
                data["res"],
                dict,
            )
        ):
            ocr_data = (
                data["res"]
            )

        else:
            ocr_data = data


        rec_texts = (
            ocr_data.get(
                "rec_texts",
                [],
            )
        )


        rec_scores = (
            ocr_data.get(
                "rec_scores",
                [],
            )
        )


        rec_boxes = (
            ocr_data.get(
                "rec_boxes",
                [],
            )
        )


        for index, text in enumerate(
            rec_texts
        ):
            score = None
            box = None


            if index < len(
                rec_scores
            ):
                score = float(
                    rec_scores[
                        index
                    ]
                )


            if index < len(
                rec_boxes
            ):
                raw_box = (
                    rec_boxes[
                        index
                    ]
                )


                if hasattr(
                    raw_box,
                    "tolist",
                ):
                    box = (
                        raw_box.tolist()
                    )
                else:
                    box = list(
                        raw_box
                    )


            detected_items.append({
                "text":
                    normalize_text(
                        text
                    ),

                "confidence":
                    score,

                "box":
                    box,
            })


    # =====================================================
    # SORT OCR DETECTIONS
    # =====================================================

    detected_items.sort(
        key=lambda item: (
            get_box_center_y(
                item.get(
                    "box"
                )
            )
            or 0,

            get_box_center_x(
                item.get(
                    "box"
                )
            ),
        )
    )


    # =====================================================
    # PHYSICAL FORM ROWS
    # =====================================================

    (
        row_boundaries,
        row_detection_method,
    ) = build_row_boundaries(
        decoded_image
    )


    (
        grouped_rows,
        unmapped_items,
    ) = group_items_by_row(
        detected_items,
        row_boundaries,
    )


    # =====================================================
    # STRUCTURED FORM FIELDS
    # =====================================================

    fields = (
        map_rows_to_fields(
            grouped_rows
        )
    )


    ignored_row_items = (
        grouped_rows.get(
            2,
            [],
        )
    )


    requires_review = any(
        field[
            "requires_review"
        ]
        for field
        in fields.values()
    )


    # =====================================================
    # ROW DEBUG INFORMATION
    # =====================================================

    rows = {}


    for row_number in range(
        1,
        EXPECTED_ROW_COUNT + 1,
    ):
        top, bottom = (
            row_boundaries[
                row_number - 1
            ]
        )


        rows[
            str(row_number)
        ] = {
            "top":
                int(top),

            "bottom":
                int(bottom),

            "field":
                ROW_FIELD_MAP.get(
                    row_number
                ),

            "ignored":
                ROW_FIELD_MAP.get(
                    row_number
                ) is None,

            "ocr_items":
                grouped_rows.get(
                    row_number,
                    [],
                ),
        }


    # =====================================================
    # RESPONSE
    # =====================================================

    return {
        "status":
            "success",

        "file": {
            "filename":
                image.filename,

            "content_type":
                image.content_type,

            "size_bytes":
                len(contents),
        },

        "image": {
            "width":
                int(
                    decoded_image.shape[1]
                ),

            "height":
                int(
                    decoded_image.shape[0]
                ),
        },

        "mapping": {
            "expected_rows":
                EXPECTED_ROW_COUNT,

            "row_detection_method":
                row_detection_method,

            "ignored_rows":
                [2],
        },

        "text_count":
            len(
                detected_items
            ),

        "requires_review":
            requires_review,

        "fields":
            fields,

        "rows":
            rows,

        "ignored_row_items":
            ignored_row_items,

        "unmapped_items":
            unmapped_items,

        "raw_items":
            detected_items,
    }
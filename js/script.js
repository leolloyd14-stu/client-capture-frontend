// =========================================================
// CLIENT CAPTURE PORTAL
// Camera + Upload + Quality Control + OpenCV Processing
// + OCR Crop + FastAPI/PaddleOCR Integration
// =========================================================


// =========================================================
// API CONFIGURATION
// =========================================================

const OCR_API_URL = "http://127.0.0.1:8000/ocr";


// =========================================================
// FORM
// =========================================================

const form = document.getElementById("clientCaptureForm");
const successMessage = document.getElementById("successMessage");


function clearErrors() {
    form.querySelectorAll(".error").forEach(el => {
        el.textContent = "";
    });

    form.querySelectorAll(".invalid").forEach(el => {
        el.classList.remove("invalid");
    });
}


function showError(field, message) {
    field.classList.add("invalid");

    const group = field.closest(".form-group");
    const error = group ? group.querySelector(".error") : null;

    if (error) {
        error.textContent = message;
    }
}


form.addEventListener("submit", function (event) {
    event.preventDefault();

    clearErrors();

    successMessage.classList.remove("show");

    let valid = true;

    const requiredFields =
        form.querySelectorAll("[required]");

    requiredFields.forEach(field => {
        if (!field.value.trim()) {
            showError(
                field,
                "This field is required."
            );

            valid = false;
        }
    });


    const email =
        document.getElementById("email");

    if (
        email.value.trim() &&
        !email.checkValidity()
    ) {
        showError(
            email,
            "Enter a valid email address."
        );

        valid = false;
    }


    const issuingDate =
        document.getElementById("issuingDate");

    const expiringDate =
        document.getElementById("expiringDate");


    if (
        issuingDate.value &&
        expiringDate.value &&
        new Date(expiringDate.value) <
        new Date(issuingDate.value)
    ) {
        showError(
            expiringDate,
            "Expiring date cannot be earlier than issuing date."
        );

        valid = false;
    }


    if (!valid) {
        return;
    }


    successMessage.classList.add("show");

    successMessage.scrollIntoView({
        behavior: "smooth",
        block: "nearest"
    });
});


// =========================================================
// CAMERA ELEMENTS
// =========================================================

const openCameraBtn =
    document.getElementById("openCameraBtn");

const uploadPhotoBtn =
    document.getElementById("uploadPhotoBtn");

const photoUploadInput =
    document.getElementById("photoUploadInput");

const cameraWorkspace =
    document.getElementById("cameraWorkspace");

const cameraPreview =
    document.getElementById("cameraPreview");

const captureCanvas =
    document.getElementById("captureCanvas");

const capturedImage =
    document.getElementById("capturedImage");

const capturePhotoBtn =
    document.getElementById("capturePhotoBtn");

const retakePhotoBtn =
    document.getElementById("retakePhotoBtn");

const usePhotoBtn =
    document.getElementById("usePhotoBtn");

const closeCameraBtn =
    document.getElementById("closeCameraBtn");

const cameraStatus =
    document.getElementById("cameraStatus");


// =========================================================
// QUALITY ELEMENTS
// =========================================================

const qualityPanel =
    document.getElementById("qualityPanel");

const qualityBadge =
    document.getElementById("qualityBadge");

const resolutionResult =
    document.getElementById("resolutionResult");

const sharpnessResult =
    document.getElementById("sharpnessResult");

const brightnessResult =
    document.getElementById("brightnessResult");

const exposureResult =
    document.getElementById("exposureResult");

const qualitySummary =
    document.getElementById("qualitySummary");


// =========================================================
// DOCUMENT PROCESSING ELEMENTS
// =========================================================

const documentProcessingPanel =
    document.getElementById(
        "documentProcessingPanel"
    );

const documentProcessingBadge =
    document.getElementById(
        "documentProcessingBadge"
    );

const sourceDocumentPreview =
    document.getElementById(
        "sourceDocumentPreview"
    );

const processedDocumentCanvas =
    document.getElementById(
        "processedDocumentCanvas"
    );

const documentProcessingSummary =
    document.getElementById(
        "documentProcessingSummary"
    );


// =========================================================
// APPLICATION STATE
// =========================================================

let cameraStream = null;

let capturedImageBlob = null;

let processedDocumentBlob = null;

let fullCorrectedDocumentBlob = null;

let documentProcessingPassed = false;

let ocrRequestInProgress = false;


// =========================================================
// IMAGE QUALITY CONFIGURATION
// =========================================================

const IMAGE_QUALITY = {
    minWidth: 1200,
    minHeight: 800,

    minSharpness: 12,

    minBrightness: 55,
    maxBrightness: 225,

    overexposedPixelLevel: 245,
    maxOverexposedRatio: 0.12,

    sampleMaxDimension: 900
};


// =========================================================
// OCR CROP CONFIGURATION
// =========================================================

const OCR_REGION = {
    leftCropRatio: 0.20,
    topCropRatio: 0.00,
    rightCropRatio: 0.00,
    bottomCropRatio: 0.00
};


// =========================================================
// STATUS
// =========================================================

function setCameraStatus(message, type = "") {
    cameraStatus.textContent = message;

    cameraStatus.classList.remove(
        "error-status",
        "success-status"
    );

    if (type === "error") {
        cameraStatus.classList.add(
            "error-status"
        );
    }

    if (type === "success") {
        cameraStatus.classList.add(
            "success-status"
        );
    }
}


// =========================================================
// QUALITY RESET
// =========================================================

function resetQualityPanel() {
    qualityPanel.hidden = true;

    qualityBadge.textContent =
        "NOT CHECKED";

    qualityBadge.className =
        "quality-badge";

    [
        resolutionResult,
        sharpnessResult,
        brightnessResult,
        exposureResult
    ].forEach(element => {
        element.textContent = "—";
        element.className = "";
    });

    qualitySummary.textContent = "";

    qualitySummary.className =
        "quality-summary";

    usePhotoBtn.disabled = true;
}


// =========================================================
// QUALITY RESULT
// =========================================================

function setResult(
    element,
    passed,
    text
) {
    element.textContent =
        `${passed ? "✓" : "✕"} ${text}`;

    element.className =
        passed ? "pass" : "fail";
}


// =========================================================
// SAMPLE IMAGE
// =========================================================

function sampledData(canvas) {
    const sample =
        document.createElement("canvas");

    const longest =
        Math.max(
            canvas.width,
            canvas.height
        );

    const scale =
        Math.min(
            1,
            IMAGE_QUALITY.sampleMaxDimension /
            longest
        );

    sample.width =
        Math.max(
            1,
            Math.round(
                canvas.width * scale
            )
        );

    sample.height =
        Math.max(
            1,
            Math.round(
                canvas.height * scale
            )
        );

    const context =
        sample.getContext(
            "2d",
            {
                willReadFrequently: true
            }
        );

    context.drawImage(
        canvas,
        0,
        0,
        sample.width,
        sample.height
    );

    return {
        data: context.getImageData(
            0,
            0,
            sample.width,
            sample.height
        ),

        width: sample.width,
        height: sample.height
    };
}


// =========================================================
// BRIGHTNESS / OVEREXPOSURE
// =========================================================

function brightnessExposure(imageData) {
    const pixels =
        imageData.data;

    let total = 0;
    let overexposed = 0;

    const count =
        pixels.length / 4;

    for (
        let i = 0;
        i < pixels.length;
        i += 4
    ) {
        const red =
            pixels[i];

        const green =
            pixels[i + 1];

        const blue =
            pixels[i + 2];


        const luminance =
            (0.2126 * red) +
            (0.7152 * green) +
            (0.0722 * blue);


        total += luminance;


        if (
            red >=
                IMAGE_QUALITY
                    .overexposedPixelLevel &&

            green >=
                IMAGE_QUALITY
                    .overexposedPixelLevel &&

            blue >=
                IMAGE_QUALITY
                    .overexposedPixelLevel
        ) {
            overexposed++;
        }
    }


    return {
        average:
            total / count,

        ratio:
            overexposed / count
    };
}


// =========================================================
// SHARPNESS
// =========================================================

function sharpnessScore(
    imageData,
    width,
    height
) {
    const pixels =
        imageData.data;

    const grayscale =
        new Float32Array(
            width * height
        );


    for (
        let i = 0, p = 0;
        i < pixels.length;
        i += 4, p++
    ) {
        grayscale[p] =
            (0.299 * pixels[i]) +
            (0.587 * pixels[i + 1]) +
            (0.114 * pixels[i + 2]);
    }


    let sum = 0;
    let sumSquares = 0;
    let count = 0;


    for (
        let y = 1;
        y < height - 1;
        y++
    ) {
        for (
            let x = 1;
            x < width - 1;
            x++
        ) {
            const index =
                y * width + x;


            const laplacian =
                (4 * grayscale[index]) -
                grayscale[index - 1] -
                grayscale[index + 1] -
                grayscale[index - width] -
                grayscale[index + width];


            sum += laplacian;

            sumSquares +=
                laplacian * laplacian;

            count++;
        }
    }


    if (!count) {
        return 0;
    }


    const mean =
        sum / count;


    return Math.sqrt(
        Math.max(
            0,

            (sumSquares / count) -
            (mean * mean)
        )
    );
}


// =========================================================
// ANALYZE IMAGE QUALITY
// =========================================================

function analyzeImageQuality(canvas) {
    const sample =
        sampledData(canvas);

    const light =
        brightnessExposure(
            sample.data
        );

    const sharpness =
        sharpnessScore(
            sample.data,
            sample.width,
            sample.height
        );


    const resolutionPassed =
        canvas.width >=
            IMAGE_QUALITY.minWidth &&

        canvas.height >=
            IMAGE_QUALITY.minHeight;


    const sharpnessPassed =
        sharpness >=
        IMAGE_QUALITY.minSharpness;


    const brightnessPassed =
        light.average >=
            IMAGE_QUALITY.minBrightness &&

        light.average <=
            IMAGE_QUALITY.maxBrightness;


    const exposurePassed =
        light.ratio <=
        IMAGE_QUALITY.maxOverexposedRatio;


    return {
        passed:
            resolutionPassed &&
            sharpnessPassed &&
            brightnessPassed &&
            exposurePassed,

        resolution: {
            passed:
                resolutionPassed,

            width:
                canvas.width,

            height:
                canvas.height
        },

        sharpness: {
            passed:
                sharpnessPassed,

            score:
                sharpness
        },

        brightness: {
            passed:
                brightnessPassed,

            average:
                light.average
        },

        exposure: {
            passed:
                exposurePassed,

            ratio:
                light.ratio
        }
    };
}


// =========================================================
// DISPLAY IMAGE QUALITY
// =========================================================

function displayQuality(result) {
    qualityPanel.hidden = false;


    setResult(
        resolutionResult,
        result.resolution.passed,
        `${result.resolution.width} × ${result.resolution.height}`
    );


    setResult(
        sharpnessResult,
        result.sharpness.passed,

        result.sharpness.passed
            ? `Good (${result.sharpness.score.toFixed(1)})`
            : `Too blurry (${result.sharpness.score.toFixed(1)})`
    );


    const brightness =
        result.brightness.average;


    let brightnessText;


    if (
        brightness <
        IMAGE_QUALITY.minBrightness
    ) {
        brightnessText =
            `Too dark (${brightness.toFixed(0)})`;
    } else if (
        brightness >
        IMAGE_QUALITY.maxBrightness
    ) {
        brightnessText =
            `Too bright (${brightness.toFixed(0)})`;
    } else {
        brightnessText =
            `Good (${brightness.toFixed(0)})`;
    }


    setResult(
        brightnessResult,
        result.brightness.passed,
        brightnessText
    );


    setResult(
        exposureResult,
        result.exposure.passed,

        result.exposure.passed
            ? `Good (${(
                result.exposure.ratio * 100
            ).toFixed(1)}%)`

            : `Possible glare (${(
                result.exposure.ratio * 100
            ).toFixed(1)}%)`
    );


    qualityBadge.textContent =
        result.passed
            ? "OCR ELIGIBLE"
            : "RETAKE REQUIRED";


    qualityBadge.className =
        `quality-badge ${
            result.passed
                ? "pass"
                : "fail"
        }`;


    qualitySummary.className =
        `quality-summary ${
            result.passed
                ? "pass"
                : "fail"
        }`;


    if (result.passed) {
        qualitySummary.textContent =
            "Image passed all four quality gates and can proceed to OCR.";
    } else {
        const fixes = [];


        if (
            !result.resolution.passed
        ) {
            fixes.push(
                "move closer to the document"
            );
        }


        if (
            !result.sharpness.passed
        ) {
            fixes.push(
                "hold the camera steady and allow it to focus"
            );
        }


        if (
            !result.brightness.passed
        ) {
            fixes.push(
                brightness <
                    IMAGE_QUALITY
                        .minBrightness

                    ? "increase the lighting"

                    : "reduce direct light"
            );
        }


        if (
            !result.exposure.passed
        ) {
            fixes.push(
                "change the camera angle to reduce glare"
            );
        }


        qualitySummary.textContent =
            `Image is not ready for OCR. Please ${fixes.join("; ")} and retake it.`;
    }


    usePhotoBtn.disabled =
        !result.passed;
}


// =========================================================
// DOCUMENT RESET
// =========================================================

function resetDocumentProcessing() {
    documentProcessingPanel.hidden =
        true;

    documentProcessingBadge.textContent =
        "NOT PROCESSED";

    documentProcessingBadge.className =
        "quality-badge";

    documentProcessingSummary.textContent =
        "";

    documentProcessingSummary.className =
        "quality-summary";

    sourceDocumentPreview.removeAttribute(
        "src"
    );

    processedDocumentCanvas.width = 0;
    processedDocumentCanvas.height = 0;

    processedDocumentBlob = null;

    fullCorrectedDocumentBlob = null;

    documentProcessingPassed = false;

    window.capturedDocumentImage = null;

    window.fullCorrectedDocumentImage =
        null;
}


// =========================================================
// OPENCV READY
// =========================================================

function waitForOpenCv(
    timeoutMs = 12000
) {
    return new Promise(
        (resolve, reject) => {
            const started =
                Date.now();


            const poll = () => {
                if (
                    window.cv &&
                    typeof cv.imread ===
                        "function" &&

                    typeof cv.findContours ===
                        "function"
                ) {
                    resolve();
                    return;
                }


                if (
                    Date.now() -
                    started >=
                    timeoutMs
                ) {
                    reject(
                        new Error(
                            "OpenCV.js did not become ready."
                        )
                    );

                    return;
                }


                setTimeout(
                    poll,
                    100
                );
            };


            poll();
        }
    );
}


// =========================================================
// ORDER FOUR CORNERS
// =========================================================

function orderQuadPoints(points) {
    const bySum =
        [...points].sort(
            (a, b) =>
                (a.x + a.y) -
                (b.x + b.y)
        );


    const topLeft =
        bySum[0];

    const bottomRight =
        bySum[3];


    const remaining =
        points.filter(
            point =>
                point !== topLeft &&
                point !== bottomRight
        );


    const byDifference =
        [...remaining].sort(
            (a, b) =>
                (a.y - a.x) -
                (b.y - b.x)
        );


    return {
        topLeft,

        topRight:
            byDifference[0],

        bottomRight,

        bottomLeft:
            byDifference[1]
    };
}


// =========================================================
// DISTANCE
// =========================================================

function pointDistance(a, b) {
    return Math.hypot(
        a.x - b.x,
        a.y - b.y
    );
}


// =========================================================
// MAT POINT ARRAY
// =========================================================

function matPointArray(mat) {
    const points = [];


    for (
        let i = 0;
        i < mat.rows;
        i++
    ) {
        points.push({
            x:
                mat.intPtr(
                    i,
                    0
                )[0],

            y:
                mat.intPtr(
                    i,
                    0
                )[1]
        });
    }


    return points;
}


// =========================================================
// DOCUMENT BOUNDARY DETECTION
// =========================================================

function detectLargestDocumentQuad(
    sourceMat
) {
    const maxDetectSide = 1200;


    const scale =
        Math.min(
            1,

            maxDetectSide /
            Math.max(
                sourceMat.cols,
                sourceMat.rows
            )
        );


    const small =
        new cv.Mat();

    const gray =
        new cv.Mat();

    const blurred =
        new cv.Mat();

    const edges =
        new cv.Mat();

    const contours =
        new cv.MatVector();

    const hierarchy =
        new cv.Mat();


    try {
        const targetSize =
            new cv.Size(
                Math.max(
                    1,
                    Math.round(
                        sourceMat.cols *
                        scale
                    )
                ),

                Math.max(
                    1,
                    Math.round(
                        sourceMat.rows *
                        scale
                    )
                )
            );


        cv.resize(
            sourceMat,
            small,
            targetSize,
            0,
            0,
            cv.INTER_AREA
        );


        cv.cvtColor(
            small,
            gray,
            cv.COLOR_RGBA2GRAY
        );


        cv.GaussianBlur(
            gray,
            blurred,
            new cv.Size(5, 5),
            0,
            0,
            cv.BORDER_DEFAULT
        );


        cv.Canny(
            blurred,
            edges,
            50,
            150
        );


        const kernel =
            cv.getStructuringElement(
                cv.MORPH_RECT,
                new cv.Size(5, 5)
            );


        cv.morphologyEx(
            edges,
            edges,
            cv.MORPH_CLOSE,
            kernel
        );


        kernel.delete();


        cv.findContours(
            edges,
            contours,
            hierarchy,
            cv.RETR_EXTERNAL,
            cv.CHAIN_APPROX_SIMPLE
        );


        let bestPoints = null;
        let bestArea = 0;


        const imageArea =
            small.cols *
            small.rows;


        for (
            let i = 0;
            i < contours.size();
            i++
        ) {
            const contour =
                contours.get(i);


            const perimeter =
                cv.arcLength(
                    contour,
                    true
                );


            const approximation =
                new cv.Mat();


            cv.approxPolyDP(
                contour,
                approximation,
                0.035 * perimeter,
                true
            );


            if (
                approximation.rows === 4 &&
                cv.isContourConvex(
                    approximation
                )
            ) {
                const area =
                    Math.abs(
                        cv.contourArea(
                            approximation
                        )
                    );


                if (
                    area >
                        imageArea * 0.08 &&

                    area <
                        imageArea * 0.995 &&

                    area >
                        bestArea
                ) {
                    bestArea =
                        area;

                    bestPoints =
                        matPointArray(
                            approximation
                        );
                }
            }


            approximation.delete();

            contour.delete();
        }


        if (!bestPoints) {
            return null;
        }


        const inverseScale =
            1 / scale;


        return bestPoints.map(
            point => ({
                x:
                    point.x *
                    inverseScale,

                y:
                    point.y *
                    inverseScale
            })
        );

    } finally {
        small.delete();
        gray.delete();
        blurred.delete();
        edges.delete();
        contours.delete();
        hierarchy.delete();
    }
}


// =========================================================
// PERSPECTIVE CORRECTION
// =========================================================

function perspectiveCorrect(
    sourceMat,
    quad
) {
    const ordered =
        orderQuadPoints(
            quad
        );


    const widthTop =
        pointDistance(
            ordered.topLeft,
            ordered.topRight
        );


    const widthBottom =
        pointDistance(
            ordered.bottomLeft,
            ordered.bottomRight
        );


    const heightLeft =
        pointDistance(
            ordered.topLeft,
            ordered.bottomLeft
        );


    const heightRight =
        pointDistance(
            ordered.topRight,
            ordered.bottomRight
        );


    const outputWidth =
        Math.max(
            1,
            Math.round(
                Math.max(
                    widthTop,
                    widthBottom
                )
            )
        );


    const outputHeight =
        Math.max(
            1,
            Math.round(
                Math.max(
                    heightLeft,
                    heightRight
                )
            )
        );


    const sourcePoints =
        cv.matFromArray(
            4,
            1,
            cv.CV_32FC2,
            [
                ordered.topLeft.x,
                ordered.topLeft.y,

                ordered.topRight.x,
                ordered.topRight.y,

                ordered.bottomRight.x,
                ordered.bottomRight.y,

                ordered.bottomLeft.x,
                ordered.bottomLeft.y
            ]
        );


    const destinationPoints =
        cv.matFromArray(
            4,
            1,
            cv.CV_32FC2,
            [
                0,
                0,

                outputWidth - 1,
                0,

                outputWidth - 1,
                outputHeight - 1,

                0,
                outputHeight - 1
            ]
        );


    const transform =
        cv.getPerspectiveTransform(
            sourcePoints,
            destinationPoints
        );


    const output =
        new cv.Mat();


    cv.warpPerspective(
        sourceMat,
        output,
        transform,

        new cv.Size(
            outputWidth,
            outputHeight
        ),

        cv.INTER_LINEAR,
        cv.BORDER_REPLICATE
    );


    sourcePoints.delete();

    destinationPoints.delete();

    transform.delete();


    return output;
}


// =========================================================
// OCR REGION CROP
// =========================================================

function createOcrRegionCrop(
    sourceCanvas
) {
    const left =
        Math.round(
            sourceCanvas.width *
            OCR_REGION.leftCropRatio
        );


    const top =
        Math.round(
            sourceCanvas.height *
            OCR_REGION.topCropRatio
        );


    const right =
        Math.round(
            sourceCanvas.width *
            OCR_REGION.rightCropRatio
        );


    const bottom =
        Math.round(
            sourceCanvas.height *
            OCR_REGION.bottomCropRatio
        );


    const cropWidth =
        sourceCanvas.width -
        left -
        right;


    const cropHeight =
        sourceCanvas.height -
        top -
        bottom;


    if (
        cropWidth <= 0 ||
        cropHeight <= 0
    ) {
        throw new Error(
            "OCR crop configuration produced an invalid image area."
        );
    }


    const cropCanvas =
        document.createElement(
            "canvas"
        );


    cropCanvas.width =
        cropWidth;

    cropCanvas.height =
        cropHeight;


    const context =
        cropCanvas.getContext(
            "2d",
            {
                willReadFrequently:
                    true
            }
        );


    context.drawImage(
        sourceCanvas,

        left,
        top,
        cropWidth,
        cropHeight,

        0,
        0,
        cropWidth,
        cropHeight
    );


    return cropCanvas;
}


// =========================================================
// CANVAS TO BLOB
// =========================================================

function canvasToBlob(
    canvas,
    type = "image/jpeg",
    quality = 0.94
) {
    return new Promise(
        resolve => {
            canvas.toBlob(
                resolve,
                type,
                quality
            );
        }
    );
}


// =========================================================
// DOCUMENT PROCESSING
// =========================================================

async function processDocumentGeometry(
    sourceCanvas
) {
    resetDocumentProcessing();


    documentProcessingPanel.hidden =
        false;


    documentProcessingBadge.textContent =
        "PROCESSING";


    documentProcessingSummary.textContent =
        "Detecting the document boundary and correcting perspective...";


    try {
        await waitForOpenCv();


        sourceDocumentPreview.src =
            sourceCanvas.toDataURL(
                "image/jpeg",
                0.9
            );


        const sourceMat =
            cv.imread(
                sourceCanvas
            );


        let corrected = null;


        try {
            const quad =
                detectLargestDocumentQuad(
                    sourceMat
                );


            if (!quad) {
                documentProcessingBadge.textContent =
                    "BOUNDARY NOT FOUND";


                documentProcessingBadge.className =
                    "quality-badge fail";


                documentProcessingSummary.className =
                    "quality-summary fail";


                documentProcessingSummary.textContent =
                    "The document boundary could not be detected reliably. Leave a small margin around the page, improve background contrast, or try another image.";


                documentProcessingPassed =
                    false;


                usePhotoBtn.disabled =
                    true;


                return false;
            }


            corrected =
                perspectiveCorrect(
                    sourceMat,
                    quad
                );


            const fullCorrectedCanvas =
                document.createElement(
                    "canvas"
                );


            fullCorrectedCanvas.width =
                corrected.cols;


            fullCorrectedCanvas.height =
                corrected.rows;


            cv.imshow(
                fullCorrectedCanvas,
                corrected
            );


            fullCorrectedDocumentBlob =
                await canvasToBlob(
                    fullCorrectedCanvas,
                    "image/jpeg",
                    0.94
                );


            if (
                !fullCorrectedDocumentBlob
            ) {
                throw new Error(
                    "Full corrected document could not be converted to an image."
                );
            }


            const ocrCanvas =
                createOcrRegionCrop(
                    fullCorrectedCanvas
                );


            processedDocumentCanvas.width =
                ocrCanvas.width;


            processedDocumentCanvas.height =
                ocrCanvas.height;


            const previewContext =
                processedDocumentCanvas
                    .getContext("2d");


            previewContext.clearRect(
                0,
                0,
                processedDocumentCanvas.width,
                processedDocumentCanvas.height
            );


            previewContext.drawImage(
                ocrCanvas,
                0,
                0
            );


            processedDocumentBlob =
                await canvasToBlob(
                    ocrCanvas,
                    "image/jpeg",
                    0.94
                );


            if (
                !processedDocumentBlob
            ) {
                throw new Error(
                    "OCR document could not be converted to an image."
                );
            }


            window.fullCorrectedDocumentImage =
                fullCorrectedDocumentBlob;


            window.capturedDocumentImage =
                processedDocumentBlob;


            documentProcessingPassed =
                true;


            documentProcessingBadge.textContent =
                "OCR IMAGE READY";


            documentProcessingBadge.className =
                "quality-badge pass";


            documentProcessingSummary.className =
                "quality-summary pass";


            documentProcessingSummary.textContent =
                `Document corrected and OCR region prepared at ${processedDocumentCanvas.width} × ${processedDocumentCanvas.height}.`;


            usePhotoBtn.disabled =
                false;


            return true;

        } finally {
            if (corrected) {
                corrected.delete();
            }

            sourceMat.delete();
        }

    } catch (error) {
        console.error(
            "Document processing failed:",
            error
        );


        documentProcessingPassed =
            false;


        usePhotoBtn.disabled =
            true;


        documentProcessingBadge.textContent =
            "PROCESSING FAILED";


        documentProcessingBadge.className =
            "quality-badge fail";


        documentProcessingSummary.className =
            "quality-summary fail";


        documentProcessingSummary.textContent =
            "Document correction could not be completed. Check the browser console and try again.";


        return false;
    }
}


// =========================================================
// CAMERA STREAM
// =========================================================

function closeCameraStream() {
    if (!cameraStream) {
        return;
    }


    cameraStream
        .getTracks()
        .forEach(
            track => track.stop()
        );


    cameraStream = null;

    cameraPreview.srcObject =
        null;
}


// =========================================================
// OPEN CAMERA
// =========================================================

async function openCamera() {
    setCameraStatus(
        "Opening camera... Please wait."
    );


    if (!window.isSecureContext) {
        setCameraStatus(
            "Camera access requires localhost or HTTPS.",
            "error"
        );

        return;
    }


    if (
        !navigator.mediaDevices ||
        !navigator.mediaDevices.getUserMedia
    ) {
        setCameraStatus(
            "This browser does not support camera access.",
            "error"
        );

        return;
    }


    try {
        closeCameraStream();


        try {
            cameraStream =
                await navigator.mediaDevices
                    .getUserMedia({
                        video: {
                            facingMode: {
                                ideal:
                                    "environment"
                            },

                            width: {
                                ideal:
                                    1920
                            },

                            height: {
                                ideal:
                                    1080
                            }
                        },

                        audio:
                            false
                    });

        } catch (
            preferredCameraError
        ) {
            console.warn(
                "Rear-camera request failed. Trying any available camera.",
                preferredCameraError
            );


            cameraStream =
                await navigator.mediaDevices
                    .getUserMedia({
                        video: true,
                        audio: false
                    });
        }


        cameraPreview.srcObject =
            cameraStream;


        try {
            await cameraPreview.play();
        } catch (playError) {
            console.warn(
                "Camera preview play warning:",
                playError
            );
        }


        cameraWorkspace.hidden =
            false;


        cameraPreview.hidden =
            false;


        capturedImage.hidden =
            true;


        capturePhotoBtn.hidden =
            false;


        retakePhotoBtn.hidden =
            true;


        usePhotoBtn.hidden =
            true;


        capturedImageBlob =
            null;


        resetQualityPanel();

        resetDocumentProcessing();


        const track =
            cameraStream
                .getVideoTracks()[0];


        const label =
            track && track.label
                ? track.label
                : "available camera";


        setCameraStatus(
            `Camera ready (${label}). Position the document inside the frame.`,
            "success"
        );

    } catch (error) {
        console.error(
            "Camera access failed:",
            error
        );


        let message =
            "The camera could not be started.";


        switch (error.name) {
            case "NotAllowedError":
            case "PermissionDeniedError":

                message =
                    "Camera permission was denied. Allow camera access and try again.";

                break;


            case "NotFoundError":
            case "DevicesNotFoundError":

                message =
                    "No camera was detected. You can use UPLOAD PHOTO instead.";

                break;


            case "NotReadableError":
            case "TrackStartError":

                message =
                    "The camera may already be in use by another application.";

                break;


            default:

                message =
                    `Camera could not be started (${error.name || "unknown error"}).`;
        }


        setCameraStatus(
            message,
            "error"
        );
    }
}


// =========================================================
// CAPTURE PHOTO
// =========================================================

function capturePhoto() {
    if (
        !cameraStream ||
        !cameraPreview.videoWidth ||
        !cameraPreview.videoHeight
    ) {
        setCameraStatus(
            "Camera image is not ready yet.",
            "error"
        );

        return;
    }


    captureCanvas.width =
        cameraPreview.videoWidth;


    captureCanvas.height =
        cameraPreview.videoHeight;


    const context =
        captureCanvas.getContext(
            "2d",
            {
                willReadFrequently:
                    true
            }
        );


    context.drawImage(
        cameraPreview,
        0,
        0,
        captureCanvas.width,
        captureCanvas.height
    );


    const result =
        analyzeImageQuality(
            captureCanvas
        );


    displayQuality(result);


    captureCanvas.toBlob(
        blob => {
            if (!blob) {
                setCameraStatus(
                    "Image capture failed.",
                    "error"
                );

                return;
            }


            capturedImageBlob =
                blob;


            capturedImage.src =
                URL.createObjectURL(
                    blob
                );


            cameraPreview.hidden =
                true;


            capturedImage.hidden =
                false;


            capturePhotoBtn.hidden =
                true;


            retakePhotoBtn.hidden =
                false;


            usePhotoBtn.hidden =
                false;


            setCameraStatus(
                result.passed
                    ? "Photo captured and passed the image-quality checks."
                    : "Photo captured, but one or more quality checks failed.",

                result.passed
                    ? "success"
                    : "error"
            );


            if (result.passed) {
                processDocumentGeometry(
                    captureCanvas
                );
            } else {
                resetDocumentProcessing();

                usePhotoBtn.disabled =
                    true;
            }
        },

        "image/jpeg",
        0.92
    );
}


// =========================================================
// RETAKE
// =========================================================

async function retakePhoto() {
    capturedImageBlob =
        null;


    capturedImage.removeAttribute(
        "src"
    );


    capturedImage.hidden =
        true;


    cameraPreview.hidden =
        false;


    capturePhotoBtn.hidden =
        false;


    retakePhotoBtn.hidden =
        true;


    usePhotoBtn.hidden =
        true;


    resetQualityPanel();

    resetDocumentProcessing();


    if (!cameraStream) {
        await openCamera();
    } else {
        setCameraStatus(
            "Camera ready. Capture the document again."
        );
    }
}


// =========================================================
// UPLOAD PHOTO
// =========================================================

function loadUploadedPhoto(file) {
    if (
        !file ||
        !file.type.startsWith(
            "image/"
        )
    ) {
        setCameraStatus(
            "Please select a valid image file.",
            "error"
        );

        return;
    }


    const image =
        new Image();


    const objectUrl =
        URL.createObjectURL(
            file
        );


    image.onload =
        function () {
            closeCameraStream();


            captureCanvas.width =
                image.naturalWidth;


            captureCanvas.height =
                image.naturalHeight;


            const context =
                captureCanvas.getContext(
                    "2d",
                    {
                        willReadFrequently:
                            true
                    }
                );


            context.drawImage(
                image,
                0,
                0,
                captureCanvas.width,
                captureCanvas.height
            );


            const result =
                analyzeImageQuality(
                    captureCanvas
                );


            displayQuality(
                result
            );


            capturedImageBlob =
                file;


            capturedImage.src =
                objectUrl;


            cameraWorkspace.hidden =
                false;


            cameraPreview.hidden =
                true;


            capturedImage.hidden =
                false;


            capturePhotoBtn.hidden =
                true;


            retakePhotoBtn.hidden =
                false;


            usePhotoBtn.hidden =
                false;


            setCameraStatus(
                result.passed
                    ? "Uploaded photo passed the image-quality checks."
                    : "Uploaded photo loaded, but one or more quality checks failed.",

                result.passed
                    ? "success"
                    : "error"
            );


            if (result.passed) {
                processDocumentGeometry(
                    captureCanvas
                );
            } else {
                resetDocumentProcessing();

                usePhotoBtn.disabled =
                    true;
            }
        };


    image.onerror =
        function () {
            URL.revokeObjectURL(
                objectUrl
            );


            setCameraStatus(
                "The selected image could not be opened.",
                "error"
            );
        };


    image.src =
        objectUrl;
}


// =========================================================
// OCR FORM FIELD HELPERS
// =========================================================

function clearOcrStyling() {
    form.querySelectorAll(
        ".ocr-success, .ocr-review"
    ).forEach(element => {
        element.classList.remove(
            "ocr-success",
            "ocr-review"
        );

        element.removeAttribute(
            "title"
        );
    });


    form.querySelectorAll(
        "option[data-ocr-generated='true']"
    ).forEach(option => {
        option.remove();
    });
}


function normalizeComparableText(value) {
    return String(
        value ?? ""
    )
        .trim()
        .toLowerCase()
        .normalize("NFD")
        .replace(
            /[\u0300-\u036f]/g,
            ""
        )
        .replace(
            /[^a-z0-9]/g,
            ""
        );
}


function setOcrElementValue(
    elementId,
    fieldResult
) {
    const element =
        document.getElementById(
            elementId
        );


    if (
        !element ||
        !fieldResult
    ) {
        return;
    }


    const value =
        String(
            fieldResult.value ?? ""
        );


    let requiresReview =
        Boolean(
            fieldResult.requires_review
        );


    const warnings =
        Array.isArray(
            fieldResult.warnings
        )
            ? [...fieldResult.warnings]
            : [];


    if (
        element.tagName ===
        "SELECT"
    ) {
        const target =
            normalizeComparableText(
                value
            );


        let matchingOption =
            [...element.options]
                .find(option =>
                    normalizeComparableText(
                        option.value ||
                        option.text
                    ) === target
                );


        if (
            matchingOption
        ) {
            element.value =
                matchingOption.value;
        } else if (value) {
            const generatedOption =
                document.createElement(
                    "option"
                );


            generatedOption.value =
                value;


            generatedOption.textContent =
                `${value} (OCR)`;


            generatedOption.dataset
                .ocrGenerated =
                "true";


            element.appendChild(
                generatedOption
            );


            element.value =
                value;


            requiresReview =
                true;


            warnings.push(
                "OCR value is not in the configured dropdown options."
            );
        } else {
            element.value = "";
        }

    } else {
        element.value =
            value;
    }


    element.classList.remove(
        "ocr-success",
        "ocr-review"
    );


    if (requiresReview) {
        element.classList.add(
            "ocr-review"
        );


        element.title =
            warnings.length
                ? warnings.join(" ")
                : "Please review this OCR value.";

    } else {
        element.classList.add(
            "ocr-success"
        );


        if (
            fieldResult.confidence != null
        ) {
            element.title =
                `OCR confidence: ${(fieldResult.confidence * 100).toFixed(1)}%`;
        }
    }


    element.dispatchEvent(
        new Event(
            "change",
            {
                bubbles: true
            }
        )
    );
}


// =========================================================
// POPULATE FORM
// =========================================================

function populateFormFromOcr(
    result
) {
    if (
        !result ||
        !result.fields
    ) {
        throw new Error(
            "OCR response does not contain mapped fields."
        );
    }


    clearOcrStyling();


    const fields =
        result.fields;


    setOcrElementValue(
        "ownerName",
        fields.ownerName
    );


    setOcrElementValue(
        "mobileNumber",
        fields.mobileNumber
    );


    setOcrElementValue(
        "address",
        fields.address
    );


    setOcrElementValue(
        "nationality",
        fields.nationality
    );


    setOcrElementValue(
        "idNumber",
        fields.idNumber
    );


    setOcrElementValue(
        "idType",
        fields.idType
    );


    setOcrElementValue(
        "issuingDate",
        fields.issuingDate
    );


    setOcrElementValue(
        "expiringDate",
        fields.expiringDate
    );


    setOcrElementValue(
        "union",
        fields.union
    );
}


// =========================================================
// SEND IMAGE TO FASTAPI/PADDLEOCR
// =========================================================

async function sendImageToOcr() {
    if (
        !window.capturedDocumentImage
    ) {
        setCameraStatus(
            "No OCR-ready image is available.",
            "error"
        );

        return;
    }


    if (ocrRequestInProgress) {
        return;
    }


    ocrRequestInProgress =
        true;


    usePhotoBtn.disabled =
        true;


    setCameraStatus(
        "Processing handwriting with PaddleOCR..."
    );


    const formData =
        new FormData();


    formData.append(
        "image",
        window.capturedDocumentImage,
        "document.jpg"
    );


    try {
        const response =
            await fetch(
                OCR_API_URL,
                {
                    method: "POST",
                    body: formData
                }
            );


        let result;


        try {
            result =
                await response.json();
        } catch {
            throw new Error(
                "OCR server returned an invalid response."
            );
        }


        if (!response.ok) {
            throw new Error(
                result?.detail ||
                `OCR request failed with HTTP ${response.status}.`
            );
        }


        populateFormFromOcr(
            result
        );


        const reviewCount =
            Object.values(
                result.fields || {}
            ).filter(
                field =>
                    field.requires_review
            ).length;


        if (
            result.requires_review ||
            reviewCount > 0
        ) {
            setCameraStatus(
                `OCR completed. ${reviewCount} field(s) require review.`,
                "success"
            );
        } else {
            setCameraStatus(
                "OCR completed successfully. Form fields were populated.",
                "success"
            );
        }


        form.scrollIntoView({
            behavior: "smooth",
            block: "start"
        });


        console.log(
            "OCR result:",
            result
        );

    } catch (error) {
        console.error(
            "OCR request failed:",
            error
        );


        setCameraStatus(
            `OCR request failed: ${error.message}`,
            "error"
        );

    } finally {
        ocrRequestInProgress =
            false;


        usePhotoBtn.disabled =
            !documentProcessingPassed;
    }
}


// =========================================================
// USE IMAGE
// =========================================================

async function useCapturedPhoto() {
    if (
        !capturedImageBlob ||
        usePhotoBtn.disabled
    ) {
        setCameraStatus(
            "The image must pass all quality checks before OCR.",
            "error"
        );

        return;
    }


    if (
        !documentProcessingPassed ||
        !processedDocumentBlob
    ) {
        setCameraStatus(
            "The document must be detected and corrected before OCR.",
            "error"
        );

        return;
    }


    window.originalCapturedDocumentImage =
        capturedImageBlob;


    window.fullCorrectedDocumentImage =
        fullCorrectedDocumentBlob;


    window.capturedDocumentImage =
        processedDocumentBlob;


    closeCameraStream();


    await sendImageToOcr();
}


// =========================================================
// CLOSE CAMERA
// =========================================================

function closeCamera() {
    closeCameraStream();


    cameraWorkspace.hidden =
        true;


    capturedImageBlob =
        null;


    capturedImage.removeAttribute(
        "src"
    );


    resetQualityPanel();

    resetDocumentProcessing();


    setCameraStatus(
        "Camera is not active."
    );
}


// =========================================================
// INITIALISE
// =========================================================

function initialiseCameraControls() {
    try {
        if (!openCameraBtn) {
            console.error(
                "OPEN CAMERA button was not found."
            );

            return;
        }


        openCameraBtn.addEventListener(
            "click",
            openCamera
        );


        if (
            uploadPhotoBtn &&
            photoUploadInput
        ) {
            uploadPhotoBtn.addEventListener(
                "click",
                function () {
                    photoUploadInput.value =
                        "";

                    photoUploadInput.click();
                }
            );


            photoUploadInput.addEventListener(
                "change",
                function () {
                    const file =
                        photoUploadInput.files &&
                        photoUploadInput.files[0];


                    if (file) {
                        loadUploadedPhoto(
                            file
                        );
                    }
                }
            );
        }


        if (capturePhotoBtn) {
            capturePhotoBtn.addEventListener(
                "click",
                capturePhoto
            );
        }


        if (retakePhotoBtn) {
            retakePhotoBtn.addEventListener(
                "click",
                retakePhoto
            );
        }


        if (usePhotoBtn) {
            usePhotoBtn.addEventListener(
                "click",
                useCapturedPhoto
            );
        }


        if (closeCameraBtn) {
            closeCameraBtn.addEventListener(
                "click",
                closeCamera
            );
        }


        window.addEventListener(
            "beforeunload",
            closeCameraStream
        );


        setCameraStatus(
            "Camera controls ready. Click OPEN CAMERA or UPLOAD PHOTO."
        );


        console.info(
            "Camera controls initialised successfully."
        );

    } catch (error) {
        console.error(
            "Camera control initialisation failed:",
            error
        );


        if (cameraStatus) {
            setCameraStatus(
                `Camera controls failed to initialise: ${error.message}`,
                "error"
            );
        }
    }
}


if (
    document.readyState ===
    "loading"
) {
    document.addEventListener(
        "DOMContentLoaded",
        initialiseCameraControls
    );
} else {
    initialiseCameraControls();
}
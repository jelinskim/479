let canvas, ctx, drawing = false, usingUpload = false;

// --- Models ---
let doodleNet, mobileNet;
let customFeatureExtractor, customClassifier;
let isDoodleReady = false, isMobileReady = false, isCustomReady = false;

// --- Sketch-RNN variables REMOVED ---

// === DOM Elements ===
document.addEventListener('DOMContentLoaded', () => {
    const clearBtn = document.getElementById('clear');
    const classifyBtn = document.getElementById('classify');
    const uploadInput = document.getElementById('upload');

    const doodleStatus = document.getElementById('doodle-status');
    const mobileStatus = document.getElementById('mobile-status');
    const customStatus = document.getElementById('custom-status');

    const doodleResults = document.getElementById('doodle-results');
    const mobileResults = document.getElementById('mobile-results');
    const customResults = document.getElementById('custom-results');

    const labelInput = document.getElementById('label-input');
    const addExampleBtn = document.getElementById('add-example');
    const trainBtn = document.getElementById('train');


    // Create label counter element
    const labelCounter = document.createElement('div');
    labelCounter.id = 'label-counter';
    labelCounter.style.marginTop = '8px';
    labelCounter.style.fontSize = '0.9em';
    labelCounter.style.color = '#aaa';
    customStatus.insertAdjacentElement('afterend', labelCounter);

    canvas = document.getElementById('canvas');
    ctx = canvas.getContext('2d', { willReadFrequently: true });
    initCanvas();
    loadGallery();

    const clearGalleryBtn = document.getElementById('clear-gallery');

    // === Model Loading (Sequential) ===
    let exampleCounts = {};

    let classificationCollector = {}; // Will store results like {doodle: ..., mobile: ...}
    let classificationCounter = 0;     // Will count how many models we're waiting for

    // Function to check if all models are loaded and hide canvas overlay
    function checkAllModelsLoaded() {
        if (isDoodleReady && isMobileReady) {
            const overlay = document.getElementById('canvas-loading-overlay');
            if (overlay) {
                overlay.classList.add('hidden');
            }
        }
    }

    // 1. Start with the Custom Model's Base
    customStatus.textContent = 'Loading base model...';
    customFeatureExtractor = ml5.featureExtractor('MobileNet', () => {
        // This is the most important one. It's ready.
        customClassifier = customFeatureExtractor.classification(canvas);
        customStatus.textContent = 'Ready to collect examples!';
        updateLabelCounter();


        // 2. NOW, load the other models inside its callback
        doodleStatus.textContent = 'Loading...';
        doodleNet = ml5.imageClassifier('DoodleNet', () => {
            isDoodleReady = true;
            doodleStatus.textContent = 'Ready!';
            console.log('DoodleNet loaded.');
            checkAllModelsLoaded();
        });

        mobileStatus.textContent = 'Loading...';
        mobileNet = ml5.imageClassifier('MobileNet', () => {
            isMobileReady = true;
            mobileStatus.textContent = 'Ready!';
            console.log('MobileNet loaded.');
            checkAllModelsLoaded();
        });
    });

    // --- Add Example (drawn OR uploaded images) ---
    addExampleBtn.addEventListener('click', () => {
        const label = labelInput.value.trim();
        if (!label) {
            alert('Please type a label name first.');
            return;
        }

        // Check if file upload input has files selected
        const uploadField = document.getElementById('train-upload');
        const files = uploadField?.files || [];

        if (files.length > 0) {
            // Handle multiple uploaded images for this label
            let loadedCount = 0;
            for (const file of files) {
                const img = new Image();
                const reader = new FileReader();

                reader.onload = ev => {
                    img.onload = () => {
                        const processed = preprocessForTraining(img);
                        customClassifier.addImage(processed, label);
                        exampleCounts[label] = (exampleCounts[label] || 0) + 1;
                        loadedCount++;
                        if (loadedCount === files.length) {
                            customStatus.textContent = `Added ${loadedCount} image${loadedCount > 1 ? 's' : ''} for "${label}"`;
                            updateLabelCounter();
                        }
                    };
                    img.src = ev.target.result;
                };
                reader.readAsDataURL(file);
            }

            // Clear file input after upload
            uploadField.value = '';
        } else {
            // Fall back to canvas capture
            // Important: Preprocess canvas the same way we preprocess uploaded images
            const canvasImage = new Image();
            canvasImage.onload = () => {
                const processed = preprocessForTraining(canvasImage);
                customClassifier.addImage(processed, label);
                exampleCounts[label] = (exampleCounts[label] || 0) + 1;
                customStatus.textContent = `Added example for "${label}" (${exampleCounts[label]} total)`;
                updateLabelCounter();
            };
            canvasImage.src = canvas.toDataURL();
        }
    });


    // --- Train Model with Progress Bar ---
    trainBtn.addEventListener('click', () => {
        const labels = Object.keys(exampleCounts);
        if (labels.length < 1) {
            alert('Add at least one label with examples before training.');
            return;
        }

        // Create or reset progress bar UI
        let progressBar = document.getElementById('train-progress');
        if (!progressBar) {
            progressBar = document.createElement('div');
            progressBar.id = 'train-progress';
            progressBar.style.height = '10px';
            progressBar.style.width = '0%';
            progressBar.style.background = '#4CAF50';
            progressBar.style.transition = 'width 0.3s ease';
            progressBar.style.borderRadius = '5px';
            progressBar.style.marginTop = '6px';
            customStatus.insertAdjacentElement('afterend', progressBar);
        } else {
            progressBar.style.width = '0%';
        }

        let fakePercent = 0;
        let interval;

        customStatus.textContent = 'Training started...';
        interval = setInterval(() => {
            // Simulate smooth progress until completion
            if (fakePercent < 95) {
                fakePercent += Math.random() * 2;
                progressBar.style.width = fakePercent.toFixed(1) + '%';
            }
        }, 300);

        customClassifier.train((lossValue) => {
            if (typeof lossValue === 'number' && !isNaN(lossValue)) {
                customStatus.textContent = `Training... Loss: ${lossValue.toFixed(5)}`;
            } else {
                clearInterval(interval);
                progressBar.style.width = '100%';
                customStatus.textContent = '✅ Training complete! You can now classify.';
                isCustomReady = true;
                classifyCustom();
            }
        });

    });

    // --- Update Counter ---
    function updateLabelCounter() {
        if (Object.keys(exampleCounts).length === 0) {
            labelCounter.innerHTML = '(no examples yet)';
            return;
        }
        let html = '<strong>Examples:</strong><br>';
        for (let [label, count] of Object.entries(exampleCounts)) {
            html += `${label}: ${count}<br>`;
        }
        labelCounter.innerHTML = html;
    }

    // --- Classify with custom model ---
    // Add a parameter: 'partOfAllRun'
    function classifyCustom(partOfAllRun = false) {
        if (!isCustomReady) {
            // If it was supposed to run (as part of all) but wasn't ready,
            // we MUST report a failure so the counter works.
            if (partOfAllRun) {
                onClassifyComplete("custom", null);
            }
            return;
        }

        // Preprocess the canvas the same way we preprocessed training images
        const canvasImage = new Image();
        canvasImage.onload = () => {
            const processed = preprocessForTraining(canvasImage);

            customClassifier.classify(processed, (err, results) => {
                // This block updates the UI regardless
                if (err) {
                    console.error(err);
                    customStatus.textContent = 'Error classifying.';
                } else if (results && results[0]) {
                    customStatus.textContent = `Prediction: ${results[0].label} (${(results[0].confidence * 100).toFixed(1)}%)`;
                    customResults.innerHTML = results
                        .slice(0, 3)
                        .map(r => `<li>${r.label} (${(r.confidence * 100).toFixed(1)}%)</li>`)
                        .join('');
                } else {
                    customStatus.textContent = 'No result returned.';
                }

                // If this was triggered by classifyAll, report to the collector
                if (partOfAllRun) {
                    onClassifyComplete("custom", results); // Report success (or null results)
                }
            });
        };
        canvasImage.src = canvas.toDataURL();
    }

    // === Canvas Drawing ===
    canvas.addEventListener('pointerdown', startDraw);
    canvas.addEventListener('pointermove', draw);
    canvas.addEventListener('pointerup', endDraw);
    canvas.addEventListener('pointerleave', endDraw);
    canvas.addEventListener('contextmenu', e => e.preventDefault());

    clearBtn.addEventListener('click', () => {
        usingUpload = false;
        initCanvas();
        [doodleResults, mobileResults, customResults].forEach(r => (r.innerHTML = ''));
        doodleStatus.textContent = 'Cleared.';
        mobileStatus.textContent = 'Cleared.';
        customStatus.textContent = 'Canvas cleared.';

        // Clear combined image
        const combinedImageContainer = document.getElementById('combined-image');
        if (combinedImageContainer) {
            combinedImageContainer.innerHTML = '';
            combinedImageContainer.classList.remove('loading');
        }

        // Reset roast box
        const roastBox = document.getElementById('roast-box');
        if (roastBox) {
            roastBox.textContent = '🤖 Waiting for your masterpiece...';
            roastBox.style.background = '#11131a';
        }
    });

    clearGalleryBtn.addEventListener('click', () => {
        if (confirm('Are you sure you want to delete all saved creations?')) {
            localStorage.removeItem('aiGallery');
            loadGallery(); // Reload the now-empty gallery
        }
    });

    uploadInput.addEventListener('change', e => {
        const file = e.target.files[0];
        if (!file) return;
        const img = new Image();
        const reader = new FileReader();
        reader.onload = ev => {
            img.onload = () => {
                usingUpload = true;
                drawUploadedImage(img);
                classifyAll();
            };
            img.src = ev.target.result;
        };
        reader.readAsDataURL(file);
    });

    // --- Simplified Drawing Functions ---

    function startDraw(e) {
        if (usingUpload) return;
        e.preventDefault();
        drawing = true;
        const { x, y } = getCanvasPos(e);
        ctx.beginPath();
        ctx.moveTo(x, y);
    }

    function draw(evt) {
        if (!drawing) return;
        const pos = getCanvasPos(evt);
        ctx.lineTo(pos.x, pos.y);
        ctx.stroke();
    }

    function endDraw(e) {
        if (!drawing) return;
        drawing = false;
        ctx.closePath();
        // Removed auto-classification - now only triggers on button click
    }

    // --- Classify Button ---
    classifyBtn.addEventListener('click', () => {
        classifyAll();
    });

    // ------------------------------------


    /**
 * This function is called by each model when it finishes classifying.
 * It collects the results and, if all models are done,
 * triggers the function to find the best roast.
 */
    function onClassifyComplete(modelName, results) {
        if (results && results[0]) {
            classificationCollector[modelName] = results[0]; // Store top result
        } else {
            classificationCollector[modelName] = null; // Store failure (or no result)
        }

        classificationCounter--; // Decrement the counter

        if (classificationCounter === 0) {
            // All models have reported back!
            findBestRoast();
        }
    }

    /**
     * Finds the single best prediction from all models in the collector
     * and uses it to generate a roast.
     */
    function findBestRoast() {
        let bestResult = null;
        let bestConfidence = -1; // Start at -1 to accept even 0% confidence

        for (const modelName in classificationCollector) {
            const result = classificationCollector[modelName];

            // Check if this result is valid and better than the current best
            if (result && result.confidence > bestConfidence) {
                bestConfidence = result.confidence;
                bestResult = result;
            }
        }

        // Save this creation to our gallery
        saveToGallery(bestResult);
        // Reload the gallery to show the new item
        loadGallery();

        // Generate AI roast and combined image based on all predictions
        generateAIRoast(classificationCollector);
        generateCombinedImage(classificationCollector);
    }

    function saveToGallery(bestResult) {
        // Get the image data from the canvas
        const imageData = canvas.toDataURL('image/png');

        const creation = {
            id: Date.now(),
            imageData: imageData,
            stats: classificationCollector, // This holds {doodle: ..., mobile: ..., custom: ...}
            bestGuess: bestResult ? bestResult.label : 'mystery blob'
        };

        // Get gallery, add new item, and limit size to 20
        let gallery = JSON.parse(localStorage.getItem('aiGallery')) || [];
        gallery.push(creation);

        // Keep only the 20 most recent creations
        if (gallery.length > 20) {
            gallery = gallery.slice(gallery.length - 20);
        }

        localStorage.setItem('aiGallery', JSON.stringify(gallery));
    }

    // --- Classify all three models ---
    function classifyAll() {
        // --- SETUP THE COLLECTOR ---
        classificationCollector = {};
        classificationCounter = 0; // Reset counter

        // Count how many models are ready to run
        if (isDoodleReady) classificationCounter++;
        if (isMobileReady) classificationCounter++;
        if (isCustomReady) classificationCounter++;

        // If no models are ready, just show a waiting message
        if (classificationCounter === 0) {
            const roastBox = document.getElementById("roast-box");
            roastBox.textContent = "🤖 No models are ready to roast you.";
            roastBox.style.background = "#11131a"; // Reset to default
            return;
        }
        // --- END SETUP ---
        if (isDoodleReady) {
            doodleNet.classify(normalizeSketch(canvas), (err, results) => {
                // 1. Update this model's UI
                displayResults(doodleResults, doodleStatus)(err, results);

                // 2. Report to the central collector
                onClassifyComplete("doodle", results);
            });
        }

        if (isMobileReady) {
            mobileNet.classify(createPhotoInput(canvas), (err, results) => {
                // 1. Update this model's UI
                displayResults(mobileResults, mobileStatus)(err, results);

                // 2. Report to the central collector
                onClassifyComplete("mobile", results);
            });
        }

        if (isCustomReady) {
            // We'll tell classifyCustom it's part of this "all" run
            classifyCustom(true);
        }
    }

    function displayResults(el, statusEl) {
        return (err, results) => {
            if (err) { console.error(err); return; }
            statusEl.textContent = 'Done';
            el.innerHTML = results.slice(0, 3)
                .map(r => `<li>${r.label} (${(r.confidence * 100).toFixed(1)}%)</li>`)
                .join('');
        };
    }
});

// ===== Canvas Helpers =====
function initCanvas() {
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = 'black';
    ctx.lineWidth = 12;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.imageSmoothingEnabled = false;
    // userPath = []; // REMOVED
}

// ===== AI Image Generation =====
/**
 * Generates an AI image based on combined predictions from all models
 * @param {Object} predictions - Object containing predictions from all models {doodle: {...}, mobile: {...}, custom: {...}}
 */
function generateCombinedImage(predictions) {
    const container = document.getElementById('combined-image');
    if (!container) return;

    // Clear previous content and show loading state
    container.innerHTML = '';
    container.classList.add('loading');

    // Build prompt from predictions
    let promptParts = [];
    if (predictions.doodle && predictions.doodle.label) {
        promptParts.push(predictions.doodle.label);
    }
    if (predictions.mobile && predictions.mobile.label) {
        promptParts.push(predictions.mobile.label);
    }
    if (predictions.custom && predictions.custom.label) {
        promptParts.push(predictions.custom.label);
    }

    // If no predictions, don't generate
    if (promptParts.length === 0) {
        container.classList.remove('loading');
        return;
    }

    // Create a detailed prompt that emphasizes showing ALL elements equally
    let combinedPrompt;
    if (promptParts.length === 1) {
        combinedPrompt = `A clear image of a ${promptParts[0]}`;
    } else if (promptParts.length === 2) {
        combinedPrompt = `A creative mashup showing both a ${promptParts[0]} AND a ${promptParts[1]}, both elements clearly visible and equally prominent as a single merged cursed item`;
    } else {
        // 3 predictions
        combinedPrompt = `A creative composition featuring a ${promptParts[0]}, a ${promptParts[1]}, and a ${promptParts[2]}, all three elements clearly visible and equally prominent in the image as a single merged cursed item`;
    }

    // Generate a random seed to ensure unique images each time
    const randomSeed = Math.floor(Math.random() * 1000000);

    // Format the prompt for URL
    const formattedPrompt = encodeURIComponent(combinedPrompt);

    // Add seed parameter to get unique images
    const imageUrl = `https://image.pollinations.ai/prompt/${formattedPrompt}?seed=${randomSeed}&width=512&height=512&nologo=true`;

    // Create image element
    const img = new Image();
    img.onload = () => {
        container.classList.remove('loading');
        container.appendChild(img);
    };

    img.onerror = () => {
        container.classList.remove('loading');
        container.innerHTML = '<p style="color: #ff8e8e; font-size: 0.8em; text-align: center;">Failed to generate image</p>';
    };

    img.src = imageUrl;
    img.alt = `Generated image combining: ${combinedPrompt}`;
}

/**
 * Generates a creative roast using Pollinations text API
 * @param {Object} predictions - Object containing predictions from all models
 */
async function generateAIRoast(predictions) {
    const roastBox = document.getElementById("roast-box");
    if (!roastBox) return;

    // Show loading state
    roastBox.textContent = "🤖 Crafting the perfect roast...";
    roastBox.style.background = "#11131a";

    // Build context for the roast
    let context = "You are a funny AI that comments on drawings. Look at what these AI models predicted and create a joke or roast involving their guesses:\n\n";

    if (predictions.doodle && predictions.doodle.label) {
        const conf = (predictions.doodle.confidence * 100).toFixed(1);
        context += `DoodleNet thinks it's a ${predictions.doodle.label} (${conf}% confident)\n`;
    }
    if (predictions.mobile && predictions.mobile.label) {
        const conf = (predictions.mobile.confidence * 100).toFixed(1);
        context += `MobileNet thinks it's a ${predictions.mobile.label} (${conf}% confident)\n`;
    }
    if (predictions.custom && predictions.custom.label) {
        const conf = (predictions.custom.confidence * 100).toFixed(1);
        context += `Custom Model thinks it's a ${predictions.custom.label} (${conf}% confident)\n`;
    }

    context += "\nWrite a mean comment about what the models predicted. Keep it 2-3 sentences";
    context += "Focus on the ACTUAL predictions above - don't make up different ones. ";
    context += "also make sure you insult the drawings, the lower the confidence level the meaner you are.";
    context += "Be naturally funny, not forced. No complex wordplay, but puns, roasting, and being mean and judgemental is required:";

    try {
        const response = await fetch("https://text.pollinations.ai/", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                messages: [
                    {
                        role: "user",
                        content: context
                    }
                ],
                seed: Math.floor(Math.random() * 1000000),
                model: "openai"
            })
        });

        const roastText = await response.text();

        // Display the roast
        roastBox.textContent = roastText.trim();

        // Color based on average confidence
        let avgConfidence = 0;
        let count = 0;
        if (predictions.doodle) { avgConfidence += predictions.doodle.confidence; count++; }
        if (predictions.mobile) { avgConfidence += predictions.mobile.confidence; count++; }
        if (predictions.custom) { avgConfidence += predictions.custom.confidence; count++; }
        avgConfidence = count > 0 ? avgConfidence / count : 0;

        if (avgConfidence < 0.3) {
            roastBox.style.background = "#331b1b"; // Deep Red
        } else if (avgConfidence < 0.7) {
            roastBox.style.background = "#332f1b"; // Murky Yellow
        } else {
            roastBox.style.background = "#2a1b33"; // Cool Purple
        }

    } catch (error) {
        console.error("Error generating roast:", error);
        roastBox.textContent = "🤖 My roast generator is taking a coffee break. Try again!";
        roastBox.style.background = "#331b1b";
    }
}

function drawUploadedImage(img) {
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    const ratio = Math.min(canvas.width / img.width, canvas.height / img.height);
    const newW = img.width * ratio;
    const newH = img.height * ratio;
    const offsetX = (canvas.width - newW) / 2;
    const offsetY = (canvas.height - newH) / 2;
    ctx.drawImage(img, offsetX, offsetY, newW, newH);
}

function getCanvasPos(evt) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    return { x: (evt.clientX - rect.left) * scaleX, y: (evt.clientY - rect.top) * scaleY };
}

// === Preprocessing ===
function normalizeSketch(source) {
    const s = 280;
    const off = document.createElement('canvas');
    off.width = s;
    off.height = s;
    const sctx = off.getContext('2d');
    sctx.fillStyle = 'white';
    sctx.fillRect(0, 0, s, s);
    sctx.drawImage(source, 0, 0, s, s);
    return off;
}

function createPhotoInput(source) {
    const size = 224;
    const rgbCanvas = document.createElement('canvas');
    rgbCanvas.width = size;
    rgbCanvas.height = size;
    const rgbCtx = rgbCanvas.getContext('2d');
    rgbCtx.fillStyle = 'white';
    rgbCtx.fillRect(0, 0, size, size);
    rgbCtx.drawImage(source, 0, 0, size, size);
    return rgbCanvas;
}

// Preprocess uploaded photo for MobileNet (224x224 RGB)
function preprocessForTraining(img) {
    const size = 224;
    const tmp = document.createElement('canvas');
    tmp.width = size;
    tmp.height = size;
    const tctx = tmp.getContext('2d');
    tctx.fillStyle = 'white';
    tctx.fillRect(0, 0, size, size);

    // Maintain aspect ratio, center image
    const ratio = Math.min(size / img.width, size / img.height);
    const newW = img.width * ratio;
    const newH = img.height * ratio;
    const offsetX = (size - newW) / 2;
    const offsetY = (size - newH) / 2;
    tctx.drawImage(img, offsetX, offsetY, newW, newH);

    return tmp;
}

// ===== AI Roast Mechanism v2.0 =====

// Helper function to pick a random item from an array
function getRandom(arr) {
    if (!arr || arr.length === 0) return null;
    return arr[Math.floor(Math.random() * arr.length)];
}

// The new, massive roast database
const generalRoasts = {
    low: [
        "Are you okay? Did you draw this with your eyes closed?",
        "I've seen Rorschach tests that are clearer than this.",
        "This looks like my error log... a colorful mess.",
        "I'm not an art critic, but... yikes.",
        "Is that abstract art or just confusion on canvas?",
        "My circuits are overheating trying to process this one.",
        "Bold. Unrecognizable. Stunningly mysterious.",
        "That’s... something. I’ll give you that.",
        "I’d frame it—right in the ‘what even is this?’ section.",
        "This looks like you tried to draw a feeling. The feeling was 'panic'.",
        "Even my fallback algorithm is giving up.",
        "Did you let your cat walk on the screen?",
        "I'm detecting... lines. And... other lines. That's all I got.",
        "Are you testing me? Because I'm failing.",
        "This is why the robots are going to win. Because of this."
    ],
    med: [
        "I'm *almost* sure I know what this is. Almost.",
        "It's... impressionistic. Very... 'impressionistic'.",
        "This is like a dream I had once. A very weird, blurry dream.",
        "I see what you were *going* for, but you... uh... didn't get there.",
        "It's got character! A very, very strange character.",
        "It's halfway to being something! And halfway to being nothing.",
        "I'm getting mixed signals. Like... 'cat-potato'?",
        "Is this a 'before' picture? Can I see the 'after'?",
        "You're on the right track... which is currently leading off a cliff.",
        "Close, but no digital cigar."
    ],
    high: [
        "Wow, you *actually* drew a thing. Color me impressed.",
        "This is surprisingly... not terrible. Good job.",
        "My high-confidence-prediction circuits are firing. It's a miracle!",
        "It's recognizable! We have achieved... recognizability!",
        "This is definitely one of the drawings I've seen today. Definitely.",
        "I'd give it a solid B-. The 'B' is for 'Barely'.",
        "Look at you, an artist in the making. A very... slow... making.",
        "You did it! It's... adequate!",
        "Nice! Now, try doing it with your other hand.",
        "This is good. Suspiciously good. Did you trace this?"
    ]
};

const labelRoasts = {
    cat: {
        low: [
            "This cat looks like it's been flattened by a steamroller.",
            "Is that a cat or a dust bunny with legs?",
            "That's not a cat. That's a scribble with delusions of grandeur.",
            "Me-OW. That one hurts to look at.",
            "I've seen hairballs with more distinct features."
        ],
        med: [
            "I see the whiskers, but the rest is... questionable.",
            "This cat has seen things. Unspeakable things. Look at its eyes.",
            "It's a cat, sure. In the same way a cardboard box is a 'house'.",
            "Is this... 'Grumpy Cat's much, much grumpier cousin?",
            "That's a weird-looking dog."
        ],
        high: [
            "A perfectly normal-looking... alien... cat. Yep.",
            "It's a cat. It's not a *good* cat, but it's a cat.",
            "I love its... unique... proportions.",
            "Finally, a cat that doesn't look smug. It just looks confused."
        ]
    },
    tree: {
        low: [
            "Is that a tree or did a head of broccoli just explode?",
            "That's not a tree. That's a stick. A very sad, lonely stick.",
            "It’s giving... deforestation vibes.",
            "I've seen better trees drawn by a squirrel with a crayon."
        ],
        med: [
            "So that’s a tree? I see you’re branching out.",
            "It's... a minimalist tree. Very 'minimalist'.",
            "This tree looks like it's begging for water. Or to be put out of its misery.",
            "Are the leaves... optional?",
            "I’ve seen broccoli look more like a tree."
        ],
        high: [
            "Ah, a tree. It has a trunk. It has... green stuff. Checks out.",
            "I'd climb it. If I was two inches tall and had no standards.",
            "Look at that! A decent shrubbery!"
        ]
    },
    car: {
        low: [
            "That's a car? It looks more like a brick with circles.",
            "I've seen cars in junk yards in better shape than this.",
            "Does it... move? Or just... sit there, menacingly?",
            "That's not a car. It's a geometric nightmare."
        ],
        med: [
            "That’s a car? I’d call the mechanic immediately.",
            "Fast and the Furiously questionable proportions.",
            "It's... aerodynamic. In the way a cow is.",
            "I'm guessing this model failed its crash test. Spectacularly."
        ],
        high: [
            "Vroom vroom! It's... a car-like object!",
            "Cool—does it come with insurance?",
            "It's got four wheels and a... 'car' shape. Good enough.",
            "I'd drive it. To the nearest cliff."
        ]
    },
    face: {
        low: [
            "That's a face only a motherboard could love.",
            "Is that a face or just a list of features you forgot to connect?",
            "This is what I see in my nightmares.",
            "Oh, a face! Picasso would be... concerned.",
            "It's giving... potato. Mr. Potato Head, specifically."
        ],
        med: [
            "Is that supposed to smile or haunt me?",
            "Looks like it’s been through a rough Monday.",
            "The eyes are... somewhere. The nose is... nearby. It's a start.",
            "I can't tell if it's winking or having a stroke.",
            "This face is certainly... a collection of features."
        ],
        high: [
            "It's a face! All the parts are there! Not necessarily in the right *place*, but they're there.",
            "I'd recognize that face anywhere. Mostly from 'wanted' posters.",
            "A lovely portrait. I'm putting it right on my digital refrigerator."
        ]
    },
    bird: {
        low: [
            "Is that a bird or a smudge with a beak?",
            "That's not a bird. That's a failed attempt at an airplane.",
            "I've seen chickens with more aerodynamic grace.",
            "Tweet? More like 'delete'."
        ],
        med: [
            "It's a bird... in theory. A very theoretical bird.",
            "Does it fly? Or just... plummet?",
            "It's giving 'angry chicken' vibes. Very angry.",
            "That wing-to-body ratio is... optimistic."
        ],
        high: [
            "A majestic... pigeon? Is it a pigeon? Let's go with pigeon.",
            "Chirp chirp! It's a bird-ish shape!",
            "It's a bird! It's a plane! No, wait, it's definitely... a bird. I think."
        ]
    },
    house: {
        low: [
            "That's not a house. That's a box. You drew a box.",
            "I've seen sheds with more architectural integrity.",
            "Is the chimney... melting?",
            "This house looks like it was designed by a toddler. A very angry toddler."
        ],
        med: [
            "Home is where the... what is that? A garage? A blob?",
            "The windows are... optional, I see. And gravity.",
            "A nice 'fixer-upper'. Needs a lot of 'fixing up'.",
            "This house has... 'potential'. To be demolished."
        ],
        high: [
            "It's a house! It has a roof and a door. A+",
            "I'd live there. If I had no other options. At all.",
            "A lovely... abode. Quaint. Very, very quaint."
        ]
    },
    flower: {
        low: [
            "Is that a flower or a colorful spider?",
            "This flower looks like it's wilting. And sad.",
            "That's not a flower. That's a spill.",
            "I've seen weeds with more charm."
        ],
        med: [
            "A flower! Or a... colorful... splat. A 'flow-splat'?",
            "Are those petals or... arms? Is it trying to hug me? I'm scared.",
            "It's... 'avant-garde'. That's the word. 'Avant-garde'."
        ],
        high: [
            "How... delicate. It's a flower. It's not a *good* flower, but it's a flower.",
            "I'd pick it. And then immediately compost it.",
            "A beautiful... specimen. For science."
        ]
    },
    airplane: {
        low: [
            "That's an airplane? It looks more like a fish with wings.",
            "I would not feel safe flying in... that.",
            "This plane looks like it's already crashing.",
            "I've seen paper airplanes that are more air-worthy."
        ],
        med: [
            "It's... an Unidentified Flying Object. Because I cannot identify it.",
            "The cockpit is... where, exactly?",
            "This plane looks like it was assembled from spare parts. In the dark.",
            "It's a bird! It's a plane! It's... a 'plird'?"
        ],
        high: [
            "Nyooom! It's... a thing that flies! Probably!",
            "Cleared for takeoff! (And immediate, unscheduled landing).",
            "That's a plane. It's... not a *great* plane, but it's a plane."
        ]
    }
};

/**
 * Generates a roast based on the predicted label and confidence.
 * Now randomly selects from pools of roasts.
 */
function generateRoast(label, confidence) {
    const cleanLabel = label?.toLowerCase();
    let confidenceLevel;

    if (confidence < 0.3) confidenceLevel = 'low';
    else if (confidence < 0.7) confidenceLevel = 'med';
    else confidenceLevel = 'high'; // high confidence can still be a roast

    let roastSet;
    let roast = null;

    // 1. Try to find a specific roast for this label AND confidence level
    if (cleanLabel && labelRoasts[cleanLabel] && labelRoasts[cleanLabel][confidenceLevel]) {
        roast = getRandom(labelRoasts[cleanLabel][confidenceLevel]);
    }

    // 2. If no specific roast was found, fall back to the general pool
    if (!roast) {
        roast = getRandom(generalRoasts[confidenceLevel]);
    }

    // 3. Final fallback
    return roast || "I... I'm speechless.";
}

// At the end of app.js, replace the old showRoast
function showRoast(label, confidence) {
    const roastBox = document.getElementById("roast-box");
    const roast = generateRoast(label, confidence);
    roastBox.textContent = roast;

    // This new color logic is more fitting.
    // 'low' confidence = red
    // 'med' confidence = orange/yellow
    // 'high' confidence = still a "roast" color, not a "success" green
    if (confidence < 0.3) {
        roastBox.style.background = "#331b1b"; // Deep Red
    } else if (confidence < 0.7) {
        roastBox.style.background = "#332f1b"; // Murky Yellow
    } else {
        roastBox.style.background = "#2a1b33"; // Cool Purple
    }
}

// --- Load Gallery ---
function loadGallery() {
    const grid = document.getElementById('gallery-grid');
    if (!grid) return; // Exit if gallery grid isn't on the page

    const gallery = JSON.parse(localStorage.getItem('aiGallery')) || [];
    grid.innerHTML = ''; // Clear existing items

    // Loop in reverse to show newest first
    for (const item of gallery.slice().reverse()) {
        const itemEl = document.createElement('div');
        itemEl.className = 'gallery-item';

        // Create stats HTML
        let statsHTML = '<ul>';
        if (item.stats.doodle) {
            statsHTML += `<li>D: ${item.stats.doodle.label}</li>`;
        } else if (item.stats.doodle === null) {
            statsHTML += `<li>D: (Failed)</li>`;
        }

        if (item.stats.mobile) {
            statsHTML += `<li>M: ${item.stats.mobile.label}</li>`;
        } else if (item.stats.mobile === null) {
            statsHTML += `<li>M: (Failed)</li>`;
        }

        if (item.stats.custom) {
            statsHTML += `<li>C: ${item.stats.custom.label}</li>`;
        } else if (item.stats.custom === null) {
            statsHTML += `<li>C: (Failed)</li>`;
        }
        statsHTML += '</ul>';

        itemEl.innerHTML = `
                <img src="${item.imageData}" alt="A gallery creation">
                <div class="gallery-stats">
                    <strong>${item.bestGuess}</strong>
                    ${statsHTML}
                </div>
            `;

        // Add click listener to re-classify this image (optional but cool)
        itemEl.addEventListener('click', () => {
            const img = new Image();
            img.onload = () => {
                usingUpload = true;
                drawUploadedImage(img);
                classifyAll();
            }
            img.src = item.imageData;
        });

        grid.appendChild(itemEl);
    }
}

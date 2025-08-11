document.addEventListener('DOMContentLoaded', function () {
    // --- Get DOM Elements ---
    const uploadForm = document.getElementById('uploadForm');
    const resultsPlaceholder = document.getElementById('resultsPlaceholder');
    const loaderOverlay = document.getElementById('loaderOverlay');
    const loaderText = document.getElementById('loaderText');
    const messageBoxContainer = document.getElementById('messageBoxContainer');

    // Slice Viewer Elements
    const sliceViewerSection = document.getElementById('sliceViewerSection');
    const modalitySelect = document.getElementById('modalitySelect');
    const viewSelect = document.getElementById('viewSelect');
    const sliceSlider = document.getElementById('sliceSlider');
    const sliceNumberLabel = document.getElementById('sliceNumberLabel');
    const maxSliceLabel = document.getElementById('maxSliceLabel');
    const originalSliceImg = document.getElementById('originalSliceImg');
    const segmentedSliceImg = document.getElementById('segmentedSliceImg');
    const downloadOriginalSliceBtn = document.getElementById('downloadOriginalSliceBtn');
    const downloadSegmentedSliceBtn = document.getElementById('downloadSegmentedSliceBtn');

    // Video Viewer Elements
    const videoViewerSection = document.getElementById('videoViewerSection');
    const videoModalitySelect = document.getElementById('videoModalitySelect');
    const videoViewSelect = document.getElementById('videoViewSelect');
    const generateVideoButton = document.getElementById('generateVideoButton');
    const videoPlayerContainer = document.getElementById('videoPlayerContainer');
    const segmentationVideo = document.getElementById('segmentationVideo');
    const videoLoadingText = document.getElementById('videoLoadingText');

    // Percentages Elements
    const percentagesSection = document.getElementById('percentagesSection');
    const percentageDetails = document.getElementById('percentageDetails');

    // Download Mask Elements
    const downloadMaskSection = document.getElementById('downloadMaskSection');
    const downloadFullMaskButton = document.getElementById('downloadFullMaskButton');

    // --- Global State Variables ---
    let currentMaxSlicesPerView = {"axial": 128, "coronal": 128, "sagittal": 128}; // Default
    let originalSliceDataUrl = null; // To store base64 data for download
    let segmentedSliceDataUrl = null; // To store base64 data for download

    // --- Utility Functions ---
    function showLoader(show = true, text = "Processing, please wait...") {
        loaderText.textContent = text;
        loaderOverlay.classList.toggle('hidden', !show);
    }

    function showMessage(message, type = 'info', duration = 5000) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message-box ${type}`; // type can be 'success', 'error', 'info'
        messageDiv.textContent = message;
        messageBoxContainer.appendChild(messageDiv);

        // Trigger reflow to enable CSS transition for appearing
        void messageDiv.offsetWidth; 
        messageDiv.classList.add('show');

        setTimeout(() => {
            messageDiv.classList.remove('show');
            // Remove the element after the fade-out transition completes
            messageDiv.addEventListener('transitionend', () => messageDiv.remove());
        }, duration);
    }
    
    function resetUIForNewUpload() {
        resultsPlaceholder.classList.remove('hidden');
        sliceViewerSection.classList.add('hidden');
        videoViewerSection.classList.add('hidden');
        percentagesSection.classList.add('hidden');
        downloadMaskSection.classList.add('hidden');
        videoPlayerContainer.classList.add('hidden');
        segmentationVideo.src = ''; // Clear previous video
        originalSliceImg.src = "https://placehold.co/300x300/e2e8f0/94a3b8?text=Original+Slice";
        segmentedSliceImg.src = "https://placehold.co/300x300/e2e8f0/94a3b8?text=Segmented+Slice";
        originalSliceDataUrl = null;
        segmentedSliceDataUrl = null;
        percentageDetails.innerHTML = '<p class="text-slate-500">Percentage data will appear here after segmentation.</p>';
    }

    // --- Event Listener for Form Submission ---
    uploadForm.addEventListener('submit', async function (event) {
        event.preventDefault();
        resetUIForNewUpload();
        showLoader(true, "Uploading files and segmenting tumor...");

        const formData = new FormData(uploadForm);
        try {
            const response = await fetch('/upload_and_segment', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ error: "An unknown server error occurred." }));
                throw new Error(errorData.error || `HTTP error! Status: ${response.status}`);
            }

            const result = await response.json();
            showMessage(result.message || 'Segmentation successful!', 'success');

            resultsPlaceholder.classList.add('hidden'); // Hide placeholder
            sliceViewerSection.classList.remove('hidden');
            videoViewerSection.classList.remove('hidden');
            percentagesSection.classList.remove('hidden');
            downloadMaskSection.classList.remove('hidden');

            currentMaxSlicesPerView = result.max_slices_per_view;
            updateSliceSliderLimits(); // Update slider based on current view
            sliceSlider.value = Math.floor(currentMaxSlicesPerView[viewSelect.value.toLowerCase()] / 2) || 1; // Default to middle slice
            sliceNumberLabel.textContent = sliceSlider.value;
            
            await updateSliceViewer(); // Fetch and display initial slice
            displayPercentages(result.region_percentages);

        } catch (error) {
            console.error('Upload error:', error);
            showMessage(`Error: ${error.message}`, 'error');
            resultsPlaceholder.classList.remove('hidden'); // Show placeholder again on error
        } finally {
            showLoader(false);
        }
    });

    // --- Slice Viewer Logic ---
    function updateSliceSliderLimits() {
        const currentView = viewSelect.value.toLowerCase();
        const maxSlices = currentMaxSlicesPerView[currentView] || 128;
        sliceSlider.max = maxSlices;
        maxSliceLabel.textContent = maxSlices;
        // Adjust current value if it exceeds new max, or set to 1 if maxSlices is 0
        if (parseInt(sliceSlider.value) > maxSlices) {
            sliceSlider.value = maxSlices > 0 ? maxSlices : 1;
        } else if (maxSlices === 0 && parseInt(sliceSlider.value) > 1) {
                sliceSlider.value = 1;
        }
        sliceNumberLabel.textContent = sliceSlider.value;
    }

    async function updateSliceViewer() {
        // Only proceed if section is visible (i.e., after successful segmentation)
        if (sliceViewerSection.classList.contains('hidden')) return;

        showLoader(true, "Loading slice...");
        const sliceNum = sliceSlider.value;
        const modalityIdx = modalitySelect.value;
        const view = viewSelect.value;

        // Set placeholder images while loading
        originalSliceImg.src = `https://placehold.co/300x300/e2e8f0/94a3b8?text=Loading...`;
        segmentedSliceImg.src = `https://placehold.co/300x300/e2e8f0/94a3b8?text=Loading...`;
        originalSliceDataUrl = null; // Clear previous data
        segmentedSliceDataUrl = null;


        try {
            const response = await fetch(`/get_slice?slice_num=${sliceNum}&modality_idx=${modalityIdx}&view=${view}`);
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ error: "Failed to load slice data." }));
                throw new Error(errorData.error || `HTTP error! Status: ${response.status}`);
            }
            const data = await response.json();
            originalSliceImg.src = data.original_image_url;
            segmentedSliceImg.src = data.segmented_image_url;
            originalSliceDataUrl = data.original_image_url; // Store for download
            segmentedSliceDataUrl = data.segmented_image_url; // Store for download

        } catch (error) {
            console.error('Slice loading error:', error);
            showMessage(`Error loading slice: ${error.message}`, 'error');
            originalSliceImg.src = `https://placehold.co/300x300/e2e8f0/94a3b8?text=Error`;
            segmentedSliceImg.src = `https://placehold.co/300x300/e2e8f0/94a3b8?text=Error`;
        } finally {
            showLoader(false);
        }
    }
    
    // Helper to trigger download for base64 data URLs
    function triggerBase64Download(dataUrl, filename) {
        if (!dataUrl) {
            showMessage('No image data available to download.', 'info');
            return;
        }
        const link = document.createElement('a');
        link.href = dataUrl;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }

    downloadOriginalSliceBtn.addEventListener('click', () => {
        const sliceNum = sliceSlider.value;
        const view = viewSelect.value;
        const modalityName = modalitySelect.options[modalitySelect.selectedIndex].text;
        triggerBase64Download(originalSliceDataUrl, `original_${modalityName}_${view}_slice_${sliceNum}.png`);
    });

    downloadSegmentedSliceBtn.addEventListener('click', () => {
        const sliceNum = sliceSlider.value;
        const view = viewSelect.value;
        const modalityName = modalitySelect.options[modalitySelect.selectedIndex].text;
        triggerBase64Download(segmentedSliceDataUrl, `segmented_${modalityName}_${view}_slice_${sliceNum}.png`);
    });


    sliceSlider.addEventListener('input', () => { // Update label dynamically during slide
        sliceNumberLabel.textContent = sliceSlider.value;
    });
    sliceSlider.addEventListener('change', updateSliceViewer); // Update image on release for performance
    modalitySelect.addEventListener('change', updateSliceViewer);
    viewSelect.addEventListener('change', () => {
        updateSliceSliderLimits(); // Update limits first
        updateSliceViewer();       // Then update the view
    });

    // --- Video Viewer Logic ---
    generateVideoButton.addEventListener('click', async () => {
        showLoader(true, "Generating video, this may take a few moments...");
        videoPlayerContainer.classList.remove('hidden');
        segmentationVideo.classList.add('hidden'); // Hide previous video if any
        videoLoadingText.classList.remove('hidden');
        segmentationVideo.src = ''; // Clear previous video source

        const modalityIdx = videoModalitySelect.value;
        const view = videoViewSelect.value;

        try {
            // The response will be the video file itself (blob)
            const response = await fetch(`/get_video?modality_idx=${modalityIdx}&view=${view}`);
            if (!response.ok) {
                // Try to parse error from JSON if server sends it, otherwise use status text
                let errorMsg = `HTTP error! Status: ${response.status} ${response.statusText}`;
                try {
                    const errorData = await response.json();
                    errorMsg = errorData.error || errorMsg;
                } catch (e) { /* Ignore if response is not JSON */ }
                throw new Error(errorMsg);
            }
            const videoBlob = await response.blob();
            const videoUrl = URL.createObjectURL(videoBlob);
            segmentationVideo.src = videoUrl;
            segmentationVideo.classList.remove('hidden');
            showMessage('Video generated successfully. It will play shortly.', 'success');

        } catch (error) {
            console.error('Video generation error:', error);
            showMessage(`Error generating video: ${error.message}`, 'error');
            videoPlayerContainer.classList.add('hidden'); // Hide container on error
        } finally {
            showLoader(false);
            videoLoadingText.classList.add('hidden');
        }
    });

    // --- Percentages Display Logic ---
    function displayPercentages(percentages) {
        percentageDetails.innerHTML = ''; // Clear previous content
        if (!percentages || Object.keys(percentages).length === 0) {
            percentageDetails.innerHTML = '<p class="text-slate-600">No percentage data available.</p>';
            return;
        }

        let content = '<ul class="space-y-2 text-sm">';
        const regionOrder = ["Necrotic Core", "Edema", "Enhancing Tumor"]; // Desired order

        regionOrder.forEach(region => {
            if (percentages.hasOwnProperty(region)) {
                    const percent = percentages[region];
                    content += `<li class="flex justify-between items-center">
                                <span class="text-slate-700">${region}:</span> 
                                <span class="font-semibold text-indigo-700 bg-indigo-100 px-2 py-0.5 rounded">${percent.toFixed(2)}%</span>
                                </li>`;
            }
        });
        
        // Calculate derived percentages
        const etPercent = percentages['Enhancing Tumor'] || 0;
        const ncPercent = percentages['Necrotic Core'] || 0;
        const edemaPercent = percentages['Edema'] || 0;

        const tcPercent = ncPercent + etPercent; // Tumor Core = Necrotic Core + Enhancing Tumor
        const wtPercent = ncPercent + etPercent + edemaPercent; // Whole Tumor = All three

        content += `<li class="pt-2 mt-2 border-t border-indigo-200 flex justify-between items-center">
                        <span class="text-slate-700 font-medium">Total Enhancing Tumor (ET):</span> 
                        <span class="font-bold text-indigo-700">${etPercent.toFixed(2)}%</span>
                    </li>`;
        content += `<li class="flex justify-between items-center">
                        <span class="text-slate-700 font-medium">Total Tumor Core (TC):</span> 
                        <span class="font-bold text-indigo-700">${tcPercent.toFixed(2)}%</span>
                    </li>`;
        content += `<li class="flex justify-between items-center">
                        <span class="text-slate-700 font-medium">Whole Tumor Volume (WT):</span> 
                        <span class="font-bold text-indigo-700">${wtPercent.toFixed(2)}%</span>
                    </li>`;
        
        content += '</ul>';
        percentageDetails.innerHTML = content;
    }

    // --- Download Full Mask Logic ---
    downloadFullMaskButton.addEventListener('click', () => {
        showMessage('Preparing download... Your browser will prompt you shortly.', 'info');
        // This will trigger a file download directly from the Flask endpoint
        window.location.href = '/download_mask';
    });

        // Initialize slice slider limits on page load (though they'll be updated after first segmentation)
    updateSliceSliderLimits();
});
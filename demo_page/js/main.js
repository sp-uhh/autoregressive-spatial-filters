const selector = document.getElementById('right-pane-select');
const rightPane = document.getElementById('right-pane');

// Map select values to module paths and entry-point functions
const paneModules = {
    syntheticDataset: () => import('./synthetic_dataset.js').then(mod => mod.renderSyntheticDatasetPane()),
    recordedDataset: () => import('./recorded_dataset.js').then(mod => mod.renderRecordedDatasetPane())
};

function loadRightPane(val) {
    rightPane.innerHTML = '<div style="text-align:center;">Loading...</div>'; // Optional
    if (paneModules[val]) {
        paneModules[val]();
    } else {
        rightPane.innerHTML = '<div>Invalid pane</div>';
    }
}

// Listen to selection change
selector.addEventListener('change', e => {
    loadRightPane(e.target.value);
});

// Initial load
loadRightPane(selector.value);
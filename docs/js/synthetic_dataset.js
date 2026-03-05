let syncedAudio = null;
const STORAGE_DIR = 'https://www2.informatik.uni-hamburg.de/sp/audio/publications/autoregressive-spatial-filters/data/synthetic_examples'
const JSON_STORAGE_DIR = 'data/synthetic_examples'

export async function renderSyntheticDatasetPane() {
  syncedAudio = null;
  await pauseAllMedia();

  const leftDynamic = document.getElementById('left-dynamic');
  leftDynamic.innerHTML = `
  <h3> Synthetic Dataset </h3>
  <p class="block-text"> We introduce a novel, synthetic dataset based on the Social Force Motion Model [Helbing'95] to simulate smooth, environmentally constrained speaker movement. 
  The dataset consists of two-speaker mixtures with reverberation times between 200ms and 500ms and additional diffuse noise [Habets'07] between 20-30 dB SNR.
  We use the LibriSpeech corpus [Panayotov'15] and pair utterances according to Libri2Mix [Cosentino'20].
  The dataset generation code is publicly availible on our GITHUB-REPO.
  </p>
  <p class="block-text">
    <label for="exampleSelect">Choose example index:</label>
    <select id="exampleSelect">
    </select>
    </p>
    <br><br>
    <div class="media-row">
      <video id="trajVideo" width="320" controls>
        <source id="videoSource" src="${STORAGE_DIR}/example_0/trajectory.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </div>
  `;

  const NUM_SPEAKERS = 2;
  const SSF = 'SpatialNet';
  const TST = 'BootstrapPF';
  const EXP_TYPES = [
      'input', 
      'target',
      // `${SSF.toLowerCase()}`,
      `${SSF.toLowerCase()}-${TST.toLowerCase()}`,
      `${SSF.toLowerCase()}-${TST.toLowerCase()}-ar`,
      `${SSF.toLowerCase()}-mimo-${TST.toLowerCase()}-ar`
  ];

  const EXP_CONFIG = [
    ['-', '-', '-', '-'],
    ['Oracle', '-', '-', '-'],
    // [SSF, '\u2718', 'Oracle', '\u2718'],
    [SSF, '\u2718', TST, '\u2718'],
    [SSF, '\u2718', TST, '\u2714'],
    [SSF, '\u2714', TST, '\u2714'],
  ];

  const blankSpacer = 'data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==';

  const IMG_HEIGHT = 180;
  const IMG_WIDTH = 320;
  const TABLE_WIDTH = 250;
  const tickSize = '14px';
  const exampleIds = Array.from({ length: 15 }, (_, i) => i);

  const rightPane = document.getElementById('right-pane');
  rightPane.innerHTML = `
      <div class="results-grid" id="resultsGrid"></div>
    </div>
  `;

  const select = document.getElementById('exampleSelect');
  exampleIds.forEach(i => {
    const option = document.createElement('option');
    option.value = i;
    option.textContent = i;
    select.appendChild(option);
  });
  const idx = select && select.value !== "" ? select.value : 0;
  const videoSource = document.getElementById('videoSource');
  const trackingGrid = document.getElementById("trackingGrid");
  const resultsGrid = document.getElementById("resultsGrid");
  // const resultsTable = document.getElementById("resultsTable")

  function updateAll() {
      let idx = select && select.value !== "" ? select.value : 0;


  // Audio examples table
  resultsGrid.innerHTML = '';

  // Sticky Audio titles
  let headerRow = document.createElement('div');
  headerRow.className = 'grid-header-row';

  let headerCell = document.createElement('div');
  headerCell.className = 'grid-header-cell';
  headerCell.appendChild(document.createTextNode('Method'));
  headerRow.appendChild(headerCell);

  let headerCell2 = document.createElement('div');
  headerCell2.className = 'grid-header-cell';
  headerCell2.appendChild(document.createTextNode('Tracking performance'));
  headerRow.appendChild(headerCell2);

  for(let spk=0; spk<NUM_SPEAKERS; spk++) {
    let headerCell = document.createElement('div');
    headerCell.className = 'grid-header-cell';

    // let title = document.createElement('h4');
    // title.className = 'speaker-title';
    let dot = document.createElement('span');

    // Decide color based on spk
    let colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]; // tab:blue, tab:orange, tab:green
    // Use spk-1 since arrays are 0-indexed
    dot.style.backgroundColor = colors[spk] || "#bbbbbb"; // fallback color
    let pair = document.createElement('span');
    dot.style.display = 'inline-block';
    dot.style.width = '12px';
    dot.style.height = '12px';
    dot.style.borderRadius = '50%';
    dot.style.backgroundColor = colors[spk] || "#bbbbbb";
    dot.style.margin = '0 4px';          // symmetric spacing
    dot.style.verticalAlign = 'middle';  // <-- key line

    pair.appendChild(document.createTextNode('Enhancement performance ('));
    pair.appendChild(dot);
    pair.appendChild(document.createTextNode(')'));
    headerCell.appendChild(pair);

    headerRow.appendChild(headerCell);
  }
  resultsGrid.appendChild(headerRow);

  // table content
  for(let type_idx=0; type_idx<EXP_TYPES.length; type_idx++) {
    let type = EXP_TYPES[type_idx]

    let title = document.createElement('div');
    title.className = 'sticky-header-row';

      // Create a row div
      let row = document.createElement('div');
      row.className = 'exp-row';
      
      // Add experiment columns for each speaker
      for(let spk=-2; spk<NUM_SPEAKERS; spk++) {
          let col = document.createElement('div');
          col.className = 'exp-column';

        // Create outer container for image and controls
        let container = document.createElement('div');
        container.className = 'spectrogram-with-controls';

        let controlsRow = document.createElement('div');
        controlsRow.className = 'spectrogram-controls-row';

          if (spk === -2){
            // method configuration (aligned column width)
            let imgWrapper = document.createElement('div');
            imgWrapper.className = 'img-wrapper';

            let img = document.createElement('img');
            img.alt = `typespk{type}_spktypes​pk{spk}`;

            // make the image invisible but force column width to match spectrograms
            img.style.opacity = '0';
            img.style.height = 'var(--img-row-height)';
            img.style.width = '100%';        // fill the column
            img.style.objectFit = 'contain'; // preserve aspect ratio

            img.onload = img.onerror = function() {
              synchronizeRowHeights();
              syncHeaderWidths();
            };
            img.onerror = function() {
              this.onerror = null;  // Prevent infinite loop
              this.src = blankSpacer;
              synchronizeRowHeights();
              syncHeaderWidths();
            };

            // source for the invisible placeholder image
            img.src = `${STORAGE_DIR}/example_${idx}/spatialnet-bootstrappf/trajectory.svg`;
            // img.className = "spacer-img";
            imgWrapper.appendChild(img);
            container.appendChild(imgWrapper);

            // config table
            let table = document.createElement('table');
            table.className = "metric-table";
            table.style.display = "table";  

            let header = document.createElement('tr');
            ["Enhancement", "MIMO", "Tracking", "AR"].forEach(metric => {
              let th = document.createElement('th');
              th.innerText = metric;
              header.appendChild(th);
            });
            table.appendChild(header);

            let config = document.createElement('tr');
            EXP_CONFIG[type_idx].forEach(metric => {
              let th = document.createElement('td');
              th.innerText = metric;
              if (metric === '\u2714') {
                th.style.color = 'green';
                th.style.fontSize = tickSize;
              } else if (metric === '\u2718') {
                th.style.color = 'red';
                th.style.fontSize = tickSize;
              }
              config.appendChild(th);
            });
            table.appendChild(config);

            controlsRow.appendChild(table);
            container.appendChild(controlsRow);
            } else if (spk === -1){
            // azimuth DoA tracking
            let imgWrapper = document.createElement('div');
            imgWrapper.className = 'img-wrapper spacer-wrapper';
            
            let img = document.createElement('img');
            // img.width = IMG_WIDTH;
            // img.height = IMG_HEIGHT;
            img.alt = `typespk{type}_spktypes​pk{spk}`;
            
            img.onload = img.onerror = function() {
              synchronizeRowHeights();
              syncHeaderWidths();
            };
            img.onerror = function() {
              this.onerror = null;  // Prevent infinite loop
              this.src = blankSpacer;
              synchronizeRowHeights();
              syncHeaderWidths();
            };
            if (type === 'input' || type === 'target' || type === SSF.toLowerCase()) {
              img.src = `${STORAGE_DIR}/example_${idx}/spatialnet-bootstrappf/trajectory.svg`;
              img.style.opacity = '0';
            } else {
              img.src = `${STORAGE_DIR}/example_${idx}/${type}/trajectory.svg`;
            };
            
            imgWrapper.appendChild(img);
            container.appendChild(imgWrapper);

            if (type !== 'target' && type !== 'input' && type !== SSF.toLowerCase()) {

              let colors = ["#1f77b4", "#ff7f0e"];
            
              let table = document.createElement('table');
              table.className = "metric-table";
            
              // Header
              let headerRow = document.createElement('tr');
              ["Speaker", "MAE \u2193", "ACC \u2191"].forEach(metric => {
                let th = document.createElement('th');
                th.innerText = metric;
                headerRow.appendChild(th);
              });
              table.appendChild(headerRow);
            
              Promise.all([
                fetch(`${JSON_STORAGE_DIR}/example_${idx}/${type}/localization_spk0.json`).then(r => r.json()),
                fetch(`${JSON_STORAGE_DIR}/example_${idx}/${type}/localization_spk1.json`).then(r => r.json())
              ])
              .then(([spk0, spk1]) => {
            
                let valuesRow = document.createElement('tr');
            
                /* -------- Speaker column (DOTS) -------- */
                let speakerCell = document.createElement('td');
            
                function makeDot(spk) {
                  let dot = document.createElement('span');
                  dot.style.display = 'inline-block';
                  dot.style.width = '12px';
                  dot.style.height = '12px';
                  dot.style.borderRadius = '50%';
                  dot.style.backgroundColor = colors[spk] || "#bbbbbb";
                  dot.style.verticalAlign = 'middle';
                  return dot;
                }
            
                speakerCell.appendChild(makeDot(0));
            
                let sep = document.createElement('span');
                sep.innerText = ' / ';
                speakerCell.appendChild(sep);
            
                speakerCell.appendChild(makeDot(1));
            
                valuesRow.appendChild(speakerCell);
            
                /* -------- MAE column -------- */
                let maeCell = document.createElement('td');
                maeCell.innerHTML = `
                  <span style="color:${colors[0]}">${spk0.mae.toFixed(2)}</span>
                  /
                  <span style="color:${colors[1]}">${spk1.mae.toFixed(2)}</span>
                `;
                valuesRow.appendChild(maeCell);
            
                /* -------- ACC column -------- */
                let accCell = document.createElement('td');
                accCell.innerHTML = `
                  <span style="color:${colors[0]}">${spk0.acc_10.toFixed(2)}</span>
                  /
                  <span style="color:${colors[1]}">${spk1.acc_10.toFixed(2)}</span>
                `;
                valuesRow.appendChild(accCell);
            
                table.appendChild(valuesRow);
                controlsRow.appendChild(table);
                synchronizeRowHeights();
                syncHeaderWidths();
              })
              .catch(err => {
                console.error("Metric load error:", err);
              });
            }          
               else {
            }



            container.appendChild(controlsRow);
          } else
          {

        // spectrogram
        let imgWrapper = document.createElement('div');
        imgWrapper.className = 'img-wrapper spacer-wrapper';


        let img = document.createElement('img');
        // img.width = IMG_WIDTH;
        // img.height = IMG_HEIGHT;
        img.alt = `typespk{type}_spktypes​pk{spk}`;
        img.onload = img.onerror = function() {
          synchronizeRowHeights();
          syncHeaderWidths();
        };
        img.onerror = function() {
          this.onerror = null;  // Prevent infinite loop
          this.src = blankSpacer;
          synchronizeRowHeights();
          syncHeaderWidths();
        };
        img.src = `${STORAGE_DIR}/example_${idx}/${type}/spectrogram_spk${spk}.svg`;
        imgWrapper.appendChild(img);
        container.appendChild(imgWrapper);

          // Create audio element (hidden)
          let audio = document.createElement('audio');
          audio.src = `${STORAGE_DIR}/example_${idx}/${type}/enhanced_${spk}.wav`;

          // Create play/pause button
          let playBtn = document.createElement('button');
          playBtn.textContent = '▶️'; // Unicode play symbol, or use an <img> icon
          playBtn.className = 'spectrogram-play-btn';

          // Toggle play/pause on button click
          playBtn.onclick = function() {
            if (audio.paused) {
              audio.play();
              playBtn.textContent = '⏸️'; // pause symbol
            } else {
              audio.pause();
              playBtn.textContent = '▶️';
            }
          };

          // Optional: update button state if audio ends
          audio.onended = function() {
            playBtn.textContent = '▶️';
          };

          controlsRow.appendChild(playBtn);

          // Get metrics
          if (type !== 'target') {
            fetch(`${JSON_STORAGE_DIR}/example_${idx}/${type}/enhancement_spk${spk}.json`)
            .then(response => response.json())
            .then(data => {
              const sisdrValue = data.sisdr;
              const pesqValue = data.pesq;
              const estoiValue = data.estoi;
              // console.log("SI-SDR [dB]", sisdrValue, "PESQ", pesqValue, "ESTOI", estoiValue);
              
              let table = document.createElement('table');
              table.className = "metric-table";

              let headerRow = document.createElement('tr');
              // ["SI-SDR [dB]", "PESQ [1-5]", "ESTOI [%]"].forEach(metric => {
              ["SI-SDR", "PESQ", "ESTOI"].forEach(metric => {
                let th = document.createElement('th');
                // You can either concatenate the unicode character or use createTextNode/appending
                th.innerText = metric + "\u2191";
                headerRow.appendChild(th);
              });
              table.appendChild(headerRow);
              // Create second row: metric values (SI-SDR, PESQ, STOI)
              let valuesRow1 = document.createElement('tr');
              [sisdrValue, pesqValue, estoiValue].forEach(metric => {
                let td = document.createElement('td');
                td.innerText = metric;
                valuesRow1.appendChild(td);
              });
              table.appendChild(valuesRow1);

              controlsRow.appendChild(table);
              synchronizeRowHeights();
              syncHeaderWidths();
            })
            .catch(error => {
              console.error('Error fetching JSON:', error);
            });
          } else {
            let table = document.createElement('table');
            table.className = "metric-table";
          
            let headerRow = document.createElement('tr');
            ["SI-SDR", "PESQ", "ESTOI"].forEach(metric => {
              let th = document.createElement('th');
              th.innerText = metric + "\u2191";
              headerRow.appendChild(th);
            });
            table.appendChild(headerRow);
          
            let valuesRow1 = document.createElement('tr');
            ["∞", "5.0", "100.0"].forEach(metric => {
              let td = document.createElement('td');
              td.innerText = metric;
              valuesRow1.appendChild(td);
            });
            table.appendChild(valuesRow1);
            controlsRow.appendChild(table);
          }

          container.appendChild(controlsRow);
          container.appendChild(audio);

          }
          col.appendChild(container);

          row.appendChild(col);
      }

      resultsGrid.appendChild(row);
      requestAnimationFrame(syncHeaderWidths);
      let newVideoSrc = `${STORAGE_DIR}/example_${idx}/trajectory.mp4`;
      videoSource.src = newVideoSrc;
      videoSource.parentNode.load();

    };

    let videoElem = document.getElementById('trajVideo');
let audioElems = Array.from(resultsGrid.querySelectorAll('audio'));

// Remove controls to make video unplayable on its own
videoElem.removeAttribute('controls');

// Event guards to block user-triggered play/seeking (as in previous answer)
videoElem.onplay = null;
videoElem.onseeking = null;
videoElem.onclick = null;
videoElem.onkeydown = null;

videoElem.addEventListener('play', function(e) {
  if (!syncedAudio || syncedAudio.paused) {
    e.preventDefault();
    videoElem.pause();
  }
});
videoElem.addEventListener('seeking', function(e){
  if (!syncedAudio || syncedAudio.paused) {
    e.preventDefault();
    videoElem.currentTime = 0;
  }
});
videoElem.addEventListener('click', e => {
  e.preventDefault();
  return false;
});
videoElem.addEventListener('keydown', e => {
  e.preventDefault();
  return false;
});

// Remove previous syncedAudio for this dataset
syncedAudio = null;

audioElems.forEach(audio => {
  audio.addEventListener('play', () => {
    // Pause all other audios first
    audioElems.forEach(a => { if (a !== audio && !a.paused) a.pause(); });

    // Wait for pause to actually finish (give browser an event cycle)
    setTimeout(() => {
      syncedAudio = audio;

      // Sync video to current audio and play
      videoElem.currentTime = audio.currentTime;
      if (videoElem.paused) videoElem.play();

      // (Optionally update visual indicators here)
    }, 30);
  });

  audio.addEventListener('pause', () => {
    if (!videoElem.paused) videoElem.pause();
  });

  audio.addEventListener('seeked', () => {
    videoElem.currentTime = audio.currentTime;
  });
});
  // });

  }

    window.addEventListener('load', synchronizeRowHeights);
      setTimeout(synchronizeRowHeights, 500); // Extra after metrics fill in

  function synchronizeRowHeights() {
    // For each exp-row
    document.querySelectorAll('.exp-row').forEach(function(row){
      // Collect direct exp-column children
      const columns = Array.from(row.children).filter(el => el.classList.contains('exp-column'));
      // Reset heights for measurement
      columns.forEach(col => col.style.height = '');
      // Find the tallest
      const maxHeight = Math.max(...columns.map(col => col.offsetHeight));
      // Set all columns in the row to the tallest
      columns.forEach(col => col.style.height = maxHeight + "px");
    });
  }

  function syncHeaderWidths() {
    const headerRow = document.querySelector('.grid-header-row');
    const firstRow = document.querySelector('.exp-row');
  
    if (!headerRow || !firstRow) return;
  
    const headerCells = Array.from(headerRow.children);
    const cols = Array.from(firstRow.children);
  
    if (headerCells.length !== cols.length) return;
  
    cols.forEach((col, i) => {
      const w = col.getBoundingClientRect().width;
      headerCells[i].style.flex = `0 0 ${w}px`;
      headerCells[i].style.width = `${w}px`;
    });
  }

  select.addEventListener('change', updateAll);

  updateAll();
}

async function pauseAllMedia() {
  // Pause and reset all audios
  syncedAudio = null;
  await Promise.all(Array.from(document.querySelectorAll('audio')).map(async audio => {
      // Pause returns a Promise in modern browsers
      await audio.pause();
      audio.currentTime = 0;
  }));
  // Pause and reset video if present
  const video = document.getElementById('trajVideo');
  if (video) {
      await video.pause();
      video.currentTime = 0;
  }
}

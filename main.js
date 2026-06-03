// Main JS logic for Weather & Irradiance ML Model Web App

// Math activations and helper functions
function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
}

function tanh(x) {
    return Math.tanh(x);
}

function relu(x) {
    return Math.max(0, x);
}

// Neural Network Layers
function embedding(stateId, embeddingWeights) {
    return embeddingWeights[stateId];
}

function lstmStep(x, w_ih, b_ih, b_hh, hiddenSize) {
    const h = new Array(hiddenSize);
    const gate_in = new Array(4 * hiddenSize);
    
    // We compute: gate_in = w_ih * x + b_ih + b_hh (since h_0 = 0)
    for (let i = 0; i < 4 * hiddenSize; i++) {
        let sum = b_ih[i] + b_hh[i];
        const w_row = w_ih[i];
        for (let j = 0; j < x.length; j++) {
            sum += w_row[j] * x[j];
        }
        gate_in[i] = sum;
    }
    
    // Compute gates and cell state
    for (let k = 0; k < hiddenSize; k++) {
        const i_gate = sigmoid(gate_in[k]);
        const f_gate = sigmoid(gate_in[hiddenSize + k]);
        const g_gate = tanh(gate_in[2 * hiddenSize + k]);
        const o_gate = sigmoid(gate_in[3 * hiddenSize + k]);
        
        const c_t = i_gate * g_gate; // since c_0 = 0
        h[k] = o_gate * tanh(c_t);
    }
    return h;
}

function gruStep(x, w_ih, b_ih, b_hh, hiddenSize) {
    const h = new Array(hiddenSize);
    const x_gate = new Array(3 * hiddenSize);
    
    // We compute: x_gate = w_ih * x + b_ih
    for (let i = 0; i < 3 * hiddenSize; i++) {
        let sum = b_ih[i];
        const w_row = w_ih[i];
        for (let j = 0; j < x.length; j++) {
            sum += w_row[j] * x[j];
        }
        x_gate[i] = sum;
    }
    
    // Compute gates
    for (let k = 0; k < hiddenSize; k++) {
        const r_gate = sigmoid(x_gate[k] + b_hh[k]);
        const z_gate = sigmoid(x_gate[hiddenSize + k] + b_hh[hiddenSize + k]);
        const n_gate = tanh(x_gate[2 * hiddenSize + k] + r_gate * b_hh[2 * hiddenSize + k]);
        
        h[k] = (1 - z_gate) * n_gate; // since h_0 = 0
    }
    return h;
}

function linear(x, w, b) {
    const y = new Array(w.length);
    for (let i = 0; i < w.length; i++) {
        let sum = b[i];
        const w_row = w[i];
        for (let j = 0; j < x.length; j++) {
            sum += w_row[j] * x[j];
        }
        y[i] = sum;
    }
    return y;
}

function batchNorm1d(x, w, b, rm, rv, eps = 1e-5) {
    const y = new Array(x.length);
    for (let i = 0; i < x.length; i++) {
        const std = Math.sqrt(rv[i] + eps);
        y[i] = ((x[i] - rm[i]) / std) * w[i] + b[i];
    }
    return y;
}

// Weather Model Execution (Temperature & Wind Speed)
function runWeatherModel(model, feat, stateId) {
    const emb = embedding(stateId, model.embedding);
    const input = feat.concat(emb);
    
    const lstm0 = lstmStep(input, model.lstm.l0.w_ih, model.lstm.l0.b_ih, model.lstm.l0.b_hh, 128);
    const lstm1 = lstmStep(lstm0, model.lstm.l1.w_ih, model.lstm.l1.b_ih, model.lstm.l1.b_hh, 128);
    const gru = gruStep(lstm1, model.gru.l0.w_ih, model.gru.l0.b_ih, model.gru.l0.b_hh, 128);
    
    const h0 = linear(gru, model.head.linear0.w, model.head.linear0.b).map(relu);
    const h1 = linear(h0, model.head.linear1.w, model.head.linear1.b).map(relu);
    const h2 = linear(h1, model.head.linear2.w, model.head.linear2.b).map(relu);
    const h3 = linear(h2, model.head.linear3.w, model.head.linear3.b);
    
    return sigmoid(h3[0]);
}

// Irradiance Model Execution
function runIrradianceModel(model, feat, stateId) {
    const emb = embedding(stateId, model.embedding);
    const input = feat.concat(emb);
    
    const lstm0 = lstmStep(input, model.lstm.l0.w_ih, model.lstm.l0.b_ih, model.lstm.l0.b_hh, 256);
    const lstm1 = lstmStep(lstm0, model.lstm.l1.w_ih, model.lstm.l1.b_ih, model.lstm.l1.b_hh, 256);
    const lstm2 = lstmStep(lstm1, model.lstm.l2.w_ih, model.lstm.l2.b_ih, model.lstm.l2.b_hh, 256);
    const gru = gruStep(lstm2, model.gru.l0.w_ih, model.gru.l0.b_ih, model.gru.l0.b_hh, 256);
    
    let h = linear(gru, model.head.linear0.w, model.head.linear0.b);
    h = batchNorm1d(h, model.head.bn0.w, model.head.bn0.b, model.head.bn0.rm, model.head.bn0.rv).map(relu);
    h = linear(h, model.head.linear1.w, model.head.linear1.b);
    h = batchNorm1d(h, model.head.bn1.w, model.head.bn1.b, model.head.bn1.rm, model.head.bn1.rv).map(relu);
    h = linear(h, model.head.linear2.w, model.head.linear2.b).map(relu);
    h = linear(h, model.head.linear3.w, model.head.linear3.b).map(relu);
    h = linear(h, model.head.linear4.w, model.head.linear4.b);
    
    return sigmoid(h[0]);
}

// Feature Engineering — 8 base features matching task2.py
function buildBaseFeatures(month, day, hour, rh2m) {
    const sin_month = Math.sin(2 * Math.PI * month / 12);
    const cos_month = Math.cos(2 * Math.PI * month / 12);
    const sin_day   = Math.sin(2 * Math.PI * day   / 31);
    const cos_day   = Math.cos(2 * Math.PI * day   / 31);
    const sin_hour  = Math.sin(2 * Math.PI * hour  / 24);
    const cos_hour  = Math.cos(2 * Math.PI * hour  / 24);
    const is_day    = (6 <= hour && hour <= 18) ? 1.0 : 0.0;
    return [sin_month, cos_month, sin_day, cos_day, sin_hour, cos_hour, rh2m, is_day];
}

// DOM Setup and Events
document.addEventListener("DOMContentLoaded", () => {
    // Populate states select list
    const stateSelect = document.getElementById("state");
    MODEL_DATA.states.forEach(state => {
        const option = document.createElement("option");
        option.value = state;
        option.textContent = state;
        stateSelect.appendChild(option);
    });

    // Default values
    stateSelect.value = "Karnataka";
    document.getElementById("month").value = "10";
    document.getElementById("day").value = "15";
    document.getElementById("hour").value = "16";
    document.getElementById("rh2m").value = "55";
    document.getElementById("t2m").value = "34.0";

    // Setup prediction target toggles and T2M visibility logic
    const targetRadios = document.getElementsByName("target");
    const t2mGroup = document.getElementById("t2m-group");

    function updateT2MVisibility() {
        let selectedValue = "";
        for (const radio of targetRadios) {
            if (radio.checked) {
                selectedValue = radio.value;
                break;
            }
        }
        
        if (selectedValue === "irradiance" || selectedValue === "all") {
            t2mGroup.classList.remove("hidden-input");
            t2mGroup.querySelector("input").disabled = false;
        } else {
            t2mGroup.classList.add("hidden-input");
            t2mGroup.querySelector("input").disabled = true;
        }
    }

    for (const radio of targetRadios) {
        radio.addEventListener("change", updateT2MVisibility);
    }
    updateT2MVisibility(); // Run on startup

    // Predict Button Event
    const predictBtn = document.getElementById("predict-btn");
    predictBtn.addEventListener("click", () => {
        // Retrieve inputs
        const state = stateSelect.value;
        const month = parseInt(document.getElementById("month").value, 10);
        const day = parseInt(document.getElementById("day").value, 10);
        const hour = parseInt(document.getElementById("hour").value, 10);
        const rh2m = parseFloat(document.getElementById("rh2m").value);
        const t2m = parseFloat(document.getElementById("t2m").value);

        // Validation
        if (isNaN(month) || month < 1 || month > 12) {
            alert("Please enter a valid month (1-12)");
            return;
        }
        if (isNaN(day) || day < 1 || day > 31) {
            alert("Please enter a valid day (1-31)");
            return;
        }
        if (isNaN(hour) || hour < 0 || hour > 23) {
            alert("Please enter a valid hour (0-23)");
            return;
        }
        if (isNaN(rh2m) || rh2m < 0 || rh2m > 100) {
            alert("Please enter a valid relative humidity (0-100%)");
            return;
        }

        const stateId = MODEL_DATA.states.indexOf(state);
        if (stateId === -1) {
            alert("Invalid state selected.");
            return;
        }

        // Feature Engineering
        const baseFeat = buildBaseFeatures(month, day, hour, rh2m);

        // Determine selection
        let activeTarget = "all";
        for (const radio of targetRadios) {
            if (radio.checked) {
                activeTarget = radio.value;
                break;
            }
        }

        // Show loading spinner style
        predictBtn.textContent = "Calculating...";
        predictBtn.disabled = true;

        // Perform predictions in a setTimeout to allow UI thread to update
        setTimeout(() => {
            try {
                // Calculate temperature
                const pred_temp_scaled = runWeatherModel(MODEL_DATA.temperature_model, baseFeat, stateId);
                const scaler_temp = MODEL_DATA.scalers.temperature;
                const tempVal = (pred_temp_scaled - scaler_temp.min) / scaler_temp.scale;
                const tempResult = Math.round(tempVal * 100) / 100;

                // Calculate windspeed
                const pred_wind_scaled = runWeatherModel(MODEL_DATA.windspeed_model, baseFeat, stateId);
                const scaler_wind = MODEL_DATA.scalers.windspeed;
                const windVal = (pred_wind_scaled - scaler_wind.min) / scaler_wind.scale;
                const windResult = Math.round(windVal * 100) / 100;

                // Calculate irradiance
                const scaler_t2m = MODEL_DATA.scalers.t2m;
                const t2m_scaled = t2m * scaler_t2m.scale + scaler_t2m.min;
                const irrFeat = baseFeat.concat([t2m_scaled]);
                const pred_irr_scaled = runIrradianceModel(MODEL_DATA.irradiance_model, irrFeat, stateId);
                const scaler_irr = MODEL_DATA.scalers.irradiance;
                const irrVal = (pred_irr_scaled - scaler_irr.min) / scaler_irr.scale;
                const irrResult = Math.round(irrVal * 100) / 100;

                // Render Results
                const resultsSection = document.getElementById("results-section");
                resultsSection.classList.remove("hidden-results");

                // Target card highlights
                const tempCard = document.getElementById("temp-card");
                const windCard = document.getElementById("wind-card");
                const irrCard = document.getElementById("irr-card");

                // Reset highlights
                tempCard.classList.remove("highlight-card");
                windCard.classList.remove("highlight-card");
                irrCard.classList.remove("highlight-card");

                // Set values
                document.getElementById("temp-value").textContent = tempResult + " °C";
                document.getElementById("wind-value").textContent = windResult + " m/s";
                document.getElementById("irr-value").textContent = irrResult + " W/m²";

                // Highlight selected
                if (activeTarget === "temperature") {
                    tempCard.classList.add("highlight-card");
                } else if (activeTarget === "windspeed") {
                    windCard.classList.add("highlight-card");
                } else if (activeTarget === "irradiance") {
                    irrCard.classList.add("highlight-card");
                } else {
                    // highlight all of them subtly
                    tempCard.classList.add("highlight-card");
                    windCard.classList.add("highlight-card");
                    irrCard.classList.add("highlight-card");
                }

                // Scroll results into view smoothly
                resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

            } catch (err) {
                console.error(err);
                alert("An error occurred during prediction: " + err.message);
            } finally {
                predictBtn.textContent = "Get Prediction";
                predictBtn.disabled = false;
            }
        }, 150);
    });
});

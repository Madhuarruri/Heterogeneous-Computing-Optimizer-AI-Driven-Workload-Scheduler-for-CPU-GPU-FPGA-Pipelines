// Application State
let currentUser = null;
let currentStep = 1;
let selectedWorkload = null;
let taskParams = {};
let currentTask = null;

// Application Data
const appData = {
  workloadTypes: [
    {
      id: "matrix_multiply",
      name: "Matrix Multiplication", 
      description: "Dense matrix operations (GEMM)",
      params: ["matrix_size", "precision", "batch_size"],
      typical_devices: ["CPU", "GPU"],
      icon: "⊞"
    },
    {
      id: "fft_2d",
      name: "2D FFT",
      description: "Fast Fourier Transform operations", 
      params: ["fft_size", "precision", "batch_count"],
      typical_devices: ["GPU", "FPGA"],
      icon: "〰"
    },
    {
      id: "image_processing", 
      name: "Image Processing",
      description: "Convolution, filtering, and transforms",
      params: ["image_size", "filter_type", "batch_size"],
      typical_devices: ["GPU", "FPGA"],
      icon: "🖼"
    },
    {
      id: "resnet50",
      name: "ResNet50 Inference",
      description: "Deep learning model inference",
      params: ["batch_size", "precision", "input_size"], 
      typical_devices: ["GPU"],
      icon: "🧠"
    },
    {
      id: "video_transcode",
      name: "Video Transcoding",
      description: "H.264 to H.265 conversion",
      params: ["resolution", "quality", "fps"],
      typical_devices: ["GPU", "FPGA"],
      icon: "🎬"
    }
  ],
  sampleTasks: [
    {
      id: 1,
      name: "Matrix Multiply 2048x2048",
      status: "completed",
      device: "GPU", 
      speedup: "8.2x",
      energy_saved: "45%",
      timestamp: "2025-08-27T10:30:00"
    },
    {
      id: 2,
      name: "ResNet50 Batch 32",
      status: "running", 
      device: "GPU",
      progress: 65,
      timestamp: "2025-08-27T11:15:00"
    },
    {
      id: 3,
      name: "2D FFT 1024x1024",
      status: "completed",
      device: "FPGA",
      speedup: "12.1x", 
      energy_saved: "62%",
      timestamp: "2025-08-27T09:45:00"
    }
  ],
  devices: [
    {
      name: "CPU",
      type: "Intel Xeon Gold 6248",
      cores: 20,
      status: "available",
      utilization: 35
    },
    {
      name: "GPU", 
      type: "AMD Radeon RX 7900 XTX",
      memory: "24GB",
      status: "available",
      utilization: 18
    },
    {
      name: "FPGA",
      type: "Xilinx Versal VCK190", 
      logic_cells: "900K",
      status: "simulated",
      utilization: 0
    }
  ],
  benchmarkData: [
    {
      workload: "Matrix 1024x1024",
      cpu_time: 245,
      gpu_time: 30,
      fpga_time: 79,
      cpu_energy: 100,
      gpu_energy: 85,
      fpga_energy: 45
    },
    {
      workload: "Matrix 4096x4096", 
      cpu_time: 15800,
      gpu_time: 1264,
      fpga_time: 3292,
      cpu_energy: 100,
      gpu_energy: 78,
      fpga_energy: 52
    },
    {
      workload: "2D FFT 2048x2048",
      cpu_time: 892,
      gpu_time: 131,
      fpga_time: 73,
      cpu_energy: 100,
      gpu_energy: 88,
      fpga_energy: 38
    }
  ]
};

// Authentication Functions
function login() {
  const email = document.getElementById('login-email').value;
  const password = document.getElementById('login-password').value;
  
  if (!email || !password) {
    alert('Please enter both email and password');
    return;
  }
  
  // Simulate login
  currentUser = {
    name: "Alex Chen",
    email: email,
    role: "Performance Engineer"
  };
  
  showMainApp();
}

function demoMode() {
  currentUser = {
    name: "Demo User",
    email: "demo@example.com",
    role: "Guest"
  };
  
  showMainApp();
}

function logout() {
  currentUser = null;
  const loginScreen = document.getElementById('login-screen');
  const mainApp = document.getElementById('main-app');
  
  loginScreen.classList.remove('hidden');
  mainApp.classList.add('hidden');
  
  document.getElementById('login-email').value = '';
  document.getElementById('login-password').value = '';
}

function showMainApp() {
  const loginScreen = document.getElementById('login-screen');
  const mainApp = document.getElementById('main-app');
  
  // Hide login screen and show main app
  loginScreen.style.display = 'none';
  mainApp.style.display = 'grid';
  mainApp.classList.remove('hidden');
  
  // Update user name
  const userNameElement = document.getElementById('user-name');
  if (userNameElement && currentUser) {
    userNameElement.textContent = currentUser.name;
  }
  
  // Load initial data
  loadRecentTasks();
  loadLeaderboard();
  loadDevices();
  showDashboard();
}

// Navigation Functions
function showView(viewId) {
  // Hide all views
  const views = document.querySelectorAll('.view');
  views.forEach(view => view.classList.remove('active'));
  
  // Show selected view
  const targetView = document.getElementById(viewId);
  if (targetView) {
    targetView.classList.add('active');
  }
  
  // Update sidebar active state
  const links = document.querySelectorAll('.sidebar-link');
  links.forEach(link => link.classList.remove('active'));
}

function showDashboard() {
  showView('dashboard-view');
  const dashboardLink = document.querySelector('.sidebar-link');
  if (dashboardLink) {
    dashboardLink.classList.add('active');
  }
}

function showTaskSubmission() {
  showView('task-submission-view');
  const taskLink = document.querySelectorAll('.sidebar-link')[1];
  if (taskLink) {
    taskLink.classList.add('active');
  }
  resetWizard();
  loadWorkloadOptions();
}

function showHistory() {
  showView('history-view');
  const historyLink = document.querySelectorAll('.sidebar-link')[2];
  if (historyLink) {
    historyLink.classList.add('active');
  }
  loadTaskHistory();
}

function showBenchmarks() {
  showView('benchmarks-view');
  const benchmarkLink = document.querySelectorAll('.sidebar-link')[3];
  if (benchmarkLink) {
    benchmarkLink.classList.add('active');
  }
}

function showDevices() {
  showView('devices-view');
  const devicesLink = document.querySelectorAll('.sidebar-link')[4];
  if (devicesLink) {
    devicesLink.classList.add('active');
  }
  loadDevices();
}

// Dashboard Functions
function loadRecentTasks() {
  const container = document.getElementById('recent-tasks');
  if (!container) return;
  
  container.innerHTML = '';
  
  appData.sampleTasks.forEach(task => {
    const taskElement = document.createElement('div');
    taskElement.className = 'task-item';
    taskElement.innerHTML = `
      <div>
        <div class="task-name">${task.name}</div>
        <div class="task-status">${task.device} • ${task.status}</div>
      </div>
      <div class="status ${getStatusClass(task.status)}">${task.status}</div>
    `;
    container.appendChild(taskElement);
  });
}

function loadLeaderboard() {
  const container = document.getElementById('leaderboard');
  if (!container) return;
  
  container.innerHTML = '';
  
  const topTasks = [...appData.sampleTasks]
    .filter(task => task.speedup)
    .sort((a, b) => parseFloat(b.speedup) - parseFloat(a.speedup))
    .slice(0, 5);
  
  topTasks.forEach((task, index) => {
    const taskElement = document.createElement('div');
    taskElement.className = 'task-item';
    taskElement.innerHTML = `
      <div>
        <div class="task-name">#${index + 1} ${task.name}</div>
        <div class="task-status">${task.device} • ${task.speedup} speedup</div>
      </div>
      <div class="status status--success">${task.energy_saved} energy saved</div>
    `;
    container.appendChild(taskElement);
  });
}

function getStatusClass(status) {
  switch(status.toLowerCase()) {
    case 'completed': return 'status--success';
    case 'running': return 'status--warning';
    case 'failed': return 'status--error';
    default: return 'status--info';
  }
}

// Task Submission Wizard
function resetWizard() {
  currentStep = 1;
  selectedWorkload = null;
  taskParams = {};
  
  // Reset progress indicators
  const steps = document.querySelectorAll('.progress-step');
  steps.forEach((step, index) => {
    step.classList.toggle('active', index === 0);
  });
  
  // Show first step
  const wizardSteps = document.querySelectorAll('.wizard-step');
  wizardSteps.forEach((step, index) => {
    step.classList.toggle('active', index === 0);
  });
}

function loadWorkloadOptions() {
  const container = document.getElementById('workload-grid');
  if (!container) return;
  
  container.innerHTML = '';
  
  appData.workloadTypes.forEach(workload => {
    const workloadElement = document.createElement('div');
    workloadElement.className = 'workload-card';
    workloadElement.onclick = () => selectWorkload(workload);
    workloadElement.innerHTML = `
      <div class="workload-icon" style="font-size: 2rem; text-align: center; margin-bottom: 12px;">${workload.icon}</div>
      <h3>${workload.name}</h3>
      <p>${workload.description}</p>
      <div style="margin-top: 12px; font-size: 12px; color: var(--color-text-secondary);">
        Typical devices: ${workload.typical_devices.join(', ')}
      </div>
    `;
    container.appendChild(workloadElement);
  });
}

function selectWorkload(workload) {
  selectedWorkload = workload;
  
  // Update UI
  const cards = document.querySelectorAll('.workload-card');
  cards.forEach(card => card.classList.remove('selected'));
  
  // Find the clicked card and select it
  cards.forEach(card => {
    if (card.querySelector('h3').textContent === workload.name) {
      card.classList.add('selected');
    }
  });
}

function nextStep() {
  if (currentStep === 1 && !selectedWorkload) {
    alert('Please select a workload type');
    return;
  }
  
  if (currentStep < 4) {
    currentStep++;
    updateWizardStep();
  }
  
  if (currentStep === 2) {
    loadParameterForm();
  } else if (currentStep === 4) {
    loadTaskSummary();
  }
}

function prevStep() {
  if (currentStep > 1) {
    currentStep--;
    updateWizardStep();
  }
}

function updateWizardStep() {
  // Update progress indicators
  const steps = document.querySelectorAll('.progress-step');
  steps.forEach((step, index) => {
    step.classList.toggle('active', index < currentStep);
  });
  
  // Show current step
  const wizardSteps = document.querySelectorAll('.wizard-step');
  wizardSteps.forEach((step, index) => {
    step.classList.toggle('active', index === currentStep - 1);
  });
}

function loadParameterForm() {
  const container = document.getElementById('parameter-form');
  if (!container) return;
  
  container.innerHTML = '';
  
  if (!selectedWorkload) return;
  
  // Generate parameter form based on workload type
  const paramGroup = document.createElement('div');
  paramGroup.className = 'param-group';
  paramGroup.innerHTML = `<h3>${selectedWorkload.name} Parameters</h3>`;
  
  if (selectedWorkload.id === 'matrix_multiply') {
    paramGroup.innerHTML += `
      <div class="slider-container">
        <div class="slider-label">
          <span>Matrix Size</span>
          <span id="matrix-size-value">1024</span>
        </div>
        <input type="range" class="slider" id="matrix-size" min="512" max="4096" step="256" value="1024" oninput="updateSliderValue('matrix-size', this.value)">
      </div>
      <div class="form-group">
        <label class="form-label">Precision</label>
        <select class="form-control" id="precision">
          <option value="fp32">FP32</option>
          <option value="fp16">FP16</option>
        </select>
      </div>
      <div class="slider-container">
        <div class="slider-label">
          <span>Batch Size</span>
          <span id="batch-size-value">1</span>
        </div>
        <input type="range" class="slider" id="batch-size" min="1" max="32" value="1" oninput="updateSliderValue('batch-size', this.value)">
      </div>
    `;
  } else if (selectedWorkload.id === 'fft_2d') {
    paramGroup.innerHTML += `
      <div class="slider-container">
        <div class="slider-label">
          <span>FFT Size</span>
          <span id="fft-size-value">1024</span>
        </div>
        <input type="range" class="slider" id="fft-size" min="256" max="2048" step="256" value="1024" oninput="updateSliderValue('fft-size', this.value)">
      </div>
      <div class="form-group">
        <label class="form-label">Precision</label>
        <select class="form-control" id="precision">
          <option value="fp32">FP32</option>
          <option value="fp16">FP16</option>
        </select>
      </div>
    `;
  } else if (selectedWorkload.id === 'resnet50') {
    paramGroup.innerHTML += `
      <div class="slider-container">
        <div class="slider-label">
          <span>Batch Size</span>
          <span id="batch-size-value">1</span>
        </div>
        <input type="range" class="slider" id="batch-size" min="1" max="64" value="1" oninput="updateSliderValue('batch-size', this.value)">
      </div>
      <div class="form-group">
        <label class="form-label">Input Size</label>
        <select class="form-control" id="input-size">
          <option value="224">224x224</option>
          <option value="299">299x299</option>
          <option value="512">512x512</option>
        </select>
      </div>
    `;
  } else {
    // Generic parameter form for other workload types
    paramGroup.innerHTML += `
      <div class="slider-container">
        <div class="slider-label">
          <span>Batch Size</span>
          <span id="batch-size-value">1</span>
        </div>
        <input type="range" class="slider" id="batch-size" min="1" max="32" value="1" oninput="updateSliderValue('batch-size', this.value)">
      </div>
      <div class="form-group">
        <label class="form-label">Quality</label>
        <select class="form-control" id="quality">
          <option value="high">High</option>
          <option value="medium">Medium</option>
          <option value="low">Low</option>
        </select>
      </div>
    `;
  }
  
  container.appendChild(paramGroup);
}

function updateSliderValue(sliderId, value) {
  const valueElement = document.getElementById(sliderId + '-value');
  if (valueElement) {
    valueElement.textContent = value;
  }
}

function loadTaskSummary() {
  const container = document.getElementById('task-summary');
  if (!container) return;
  
  // Collect form values
  const goalElement = document.querySelector('input[name="goal"]:checked');
  const devicePrefElement = document.querySelector('input[name="device"]:checked');
  
  const goal = goalElement ? goalElement.value : 'balanced';
  const devicePref = devicePrefElement ? devicePrefElement.value : 'auto';
  
  container.innerHTML = `
    <div class="summary-section">
      <h3>Workload</h3>
      <p>${selectedWorkload.name} - ${selectedWorkload.description}</p>
    </div>
    <div class="summary-section">
      <h3>Optimization Goal</h3>
      <p>${goal.charAt(0).toUpperCase() + goal.slice(1)}</p>
    </div>
    <div class="summary-section">
      <h3>Device Preference</h3>
      <p>${devicePref === 'auto' ? 'Auto (Recommended)' : devicePref.toUpperCase()}</p>
    </div>
  `;
}

function submitTask() {
  // Collect all task parameters
  const goalElement = document.querySelector('input[name="goal"]:checked');
  const devicePrefElement = document.querySelector('input[name="device"]:checked');
  
  const goal = goalElement ? goalElement.value : 'balanced';
  const devicePref = devicePrefElement ? devicePrefElement.value : 'auto';
  
  currentTask = {
    id: Date.now(),
    workload: selectedWorkload,
    goal: goal,
    devicePreference: devicePref,
    timestamp: new Date().toISOString(),
    status: 'submitted'
  };
  
  // Show progress modal
  showProgressModal();
  
  // Simulate task execution
  simulateTaskExecution();
}

// Task Execution Simulation
function showProgressModal() {
  const modal = document.getElementById('progress-modal');
  if (modal) {
    modal.classList.remove('hidden');
    
    const progressInfo = document.getElementById('progress-info');
    const progressFill = document.getElementById('progress-fill');
    const progressLogs = document.getElementById('progress-logs');
    
    if (progressInfo) progressInfo.textContent = 'Initializing task...';
    if (progressFill) progressFill.style.width = '0%';
    if (progressLogs) progressLogs.innerHTML = '';
  }
}

function simulateTaskExecution() {
  const steps = [
    { progress: 10, message: 'Validating input parameters...', delay: 1000 },
    { progress: 25, message: 'Profiling available devices...', delay: 1500 },
    { progress: 40, message: 'Running ML scheduler...', delay: 1000 },
    { progress: 45, message: 'Selected device: GPU (confidence: 94%)', delay: 500 },
    { progress: 60, message: 'Compiling optimized kernels...', delay: 2000 },
    { progress: 75, message: 'Executing workload on GPU...', delay: 3000 },
    { progress: 90, message: 'Collecting performance metrics...', delay: 1000 },
    { progress: 100, message: 'Task completed successfully!', delay: 500 }
  ];
  
  let stepIndex = 0;
  
  function executeStep() {
    if (stepIndex >= steps.length) {
      setTimeout(() => {
        const modal = document.getElementById('progress-modal');
        if (modal) modal.classList.add('hidden');
        showResults();
      }, 1000);
      return;
    }
    
    const step = steps[stepIndex];
    
    setTimeout(() => {
      const progressFill = document.getElementById('progress-fill');
      const progressInfo = document.getElementById('progress-info');
      const progressLogs = document.getElementById('progress-logs');
      
      if (progressFill) progressFill.style.width = step.progress + '%';
      if (progressInfo) progressInfo.textContent = step.message;
      
      if (progressLogs) {
        const logEntry = document.createElement('div');
        logEntry.className = 'log-entry';
        logEntry.textContent = `[${new Date().toLocaleTimeString()}] ${step.message}`;
        progressLogs.appendChild(logEntry);
        progressLogs.scrollTop = progressLogs.scrollHeight;
      }
      
      stepIndex++;
      executeStep();
    }, step.delay);
  }
  
  executeStep();
}

function showResults() {
  showView('task-results-view');
  
  // Generate mock results
  const results = {
    selectedDevice: 'GPU',
    cpuTime: 2840,
    gpuTime: 347,
    fpgaTime: 1200,
    speedup: 8.2,
    energySaved: 45,
    recommendation: 'GPU chosen: 8.2× faster than CPU, 45% less energy'
  };
  
  // Update results summary
  const summaryElement = document.getElementById('results-summary');
  if (summaryElement) {
    summaryElement.innerHTML = `
      <h2>Task Completed Successfully</h2>
      <p><strong>Selected Device:</strong> ${results.selectedDevice}</p>
      <p><strong>Speedup:</strong> ${results.speedup}× faster than CPU</p>
      <p><strong>Energy Savings:</strong> ${results.energySaved}%</p>
      <div class="status status--success">${results.recommendation}</div>
    `;
  }
  
  // Create performance chart
  setTimeout(() => {
    createPerformanceChart(results);
    createEnergyChart(results);
  }, 100);
}

function createPerformanceChart(results) {
  const canvas = document.getElementById('performance-chart');
  if (!canvas) return;
  
  const ctx = canvas.getContext('2d');
  
  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: ['CPU', 'GPU', 'FPGA'],
      datasets: [{
        label: 'Execution Time (ms)',
        data: [results.cpuTime, results.gpuTime, results.fpgaTime],
        backgroundColor: ['#1FB8CD', '#FFC185', '#B4413C'],
        borderColor: ['#1FB8CD', '#FFC185', '#B4413C'],
        borderWidth: 1
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        title: {
          display: true,
          text: 'Performance Comparison'
        },
        legend: {
          display: false
        }
      },
      scales: {
        y: {
          beginAtZero: true,
          title: {
            display: true,
            text: 'Execution Time (ms)'
          }
        }
      }
    }
  });
}

function createEnergyChart(results) {
  const canvas = document.getElementById('energy-chart');
  if (!canvas) return;
  
  const ctx = canvas.getContext('2d');
  
  new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: ['Energy Saved', 'Energy Used'],
      datasets: [{
        data: [results.energySaved, 100 - results.energySaved],
        backgroundColor: ['#5D878F', '#ECEBD5'],
        borderColor: ['#5D878F', '#ECEBD5'],
        borderWidth: 1
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        title: {
          display: true,
          text: 'Energy Efficiency'
        }
      }
    }
  });
}

// History and Device Management
function loadTaskHistory() {
  const container = document.getElementById('history-table-body');
  if (!container) return;
  
  container.innerHTML = '';
  
  // Add current task if exists
  let allTasks = [...appData.sampleTasks];
  if (currentTask && currentTask.status !== 'submitted') {
    allTasks.unshift(currentTask);
  }
  
  allTasks.forEach(task => {
    const row = document.createElement('tr');
    row.innerHTML = `
      <td>${task.name}</td>
      <td>${task.device}</td>
      <td><span class="status ${getStatusClass(task.status)}">${task.status}</span></td>
      <td>${task.speedup || 'N/A'}</td>
      <td>${task.energy_saved || 'N/A'}</td>
      <td>${new Date(task.timestamp).toLocaleDateString()}</td>
    `;
    container.appendChild(row);
  });
}

function loadDevices() {
  const container = document.getElementById('devices-grid');
  if (!container) return;
  
  container.innerHTML = '';
  
  appData.devices.forEach(device => {
    const deviceElement = document.createElement('div');
    deviceElement.className = 'device-card';
    deviceElement.innerHTML = `
      <div class="device-header">
        <div class="device-name">${device.name}</div>
        <div class="status ${device.status === 'available' ? 'status--success' : 'status--warning'}">${device.status}</div>
      </div>
      <div class="device-details">
        <div class="device-detail">
          <span>Type:</span>
          <span>${device.type}</span>
        </div>
        <div class="device-detail">
          <span>${device.cores ? 'Cores:' : device.memory ? 'Memory:' : 'Logic Cells:'}</span>
          <span>${device.cores || device.memory || device.logic_cells}</span>
        </div>
        <div class="device-detail">
          <span>Utilization:</span>
          <span>${device.utilization}%</span>
        </div>
      </div>
      <div class="utilization-bar">
        <div class="utilization-fill" style="width: ${device.utilization}%"></div>
      </div>
    `;
    container.appendChild(deviceElement);
  });
}

// Initialize Application
document.addEventListener('DOMContentLoaded', function() {
  // Initialize the application once DOM is loaded
  console.log('SmartCompute Optimizer loaded');
  
  // Add keyboard navigation
  document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape') {
      const modal = document.getElementById('progress-modal');
      if (modal && !modal.classList.contains('hidden')) {
        modal.classList.add('hidden');
      }
    }
  });
});
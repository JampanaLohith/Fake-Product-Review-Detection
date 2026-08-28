document.addEventListener("DOMContentLoaded", function () {
    // ----------------------------------------------------
    // 1. Dark/Light Theme Switching
    // ----------------------------------------------------
    const htmlElement = document.documentElement;
    const themeToggleBtn = document.getElementById("themeToggleBtn");
    
    // Get stored theme or default to light
    const currentTheme = localStorage.getItem("theme") || "light";
    htmlElement.setAttribute("data-theme", currentTheme);
    updateThemeButtonIcon(currentTheme);

    if (themeToggleBtn) {
        themeToggleBtn.addEventListener("click", function () {
            const activeTheme = htmlElement.getAttribute("data-theme");
            const newTheme = activeTheme === "light" ? "dark" : "light";
            htmlElement.setAttribute("data-theme", newTheme);
            localStorage.setItem("theme", newTheme);
            updateThemeButtonIcon(newTheme);
        });
    }

    function updateThemeButtonIcon(theme) {
        if (!themeToggleBtn) return;
        const icon = themeToggleBtn.querySelector("i");
        if (theme === "dark") {
            icon.className = "bi bi-sun";
            themeToggleBtn.className = "btn btn-outline-warning btn-icon-only";
        } else {
            icon.className = "bi bi-moon-stars";
            themeToggleBtn.className = "btn btn-outline-secondary btn-icon-only";
        }
    }

    // ----------------------------------------------------
    // 2. Character Counter & Form Validation (Home Page)
    // ----------------------------------------------------
    const reviewTextArea = document.getElementById("reviewTextArea");
    const charCounter = document.getElementById("charCounter");
    const predictionForm = document.getElementById("predictionForm");
    const validationFeedback = document.getElementById("validationFeedback");
    const loadingOverlay = document.getElementById("loadingOverlay");

    if (reviewTextArea && charCounter) {
        reviewTextArea.addEventListener("input", function () {
            const count = reviewTextArea.value.length;
            charCounter.textContent = `${count} / 2000 chars`;
            
            // Real-time character count badge update
            if (count > 0 && count < 15) {
                charCounter.className = "badge bg-danger-subtle text-danger py-2 px-3 fw-medium";
            } else if (count >= 15 && count <= 1800) {
                charCounter.className = "badge bg-success-subtle text-success py-2 px-3 fw-medium";
            } else if (count > 1800) {
                charCounter.className = "badge bg-warning-subtle text-warning py-2 px-3 fw-medium";
            } else {
                charCounter.className = "badge bg-secondary-subtle text-secondary py-2 px-3 fw-medium";
            }
        });
    }

    if (predictionForm) {
        predictionForm.addEventListener("submit", function (event) {
            const reviewText = reviewTextArea.value.trim();
            
            if (reviewText.length < 15) {
                event.preventDefault();
                reviewTextArea.classList.add("is-invalid");
                validationFeedback.style.display = "block";
            } else {
                reviewTextArea.classList.remove("is-invalid");
                if (validationFeedback) validationFeedback.style.display = "none";
                // Show loading overlay
                if (loadingOverlay) loadingOverlay.classList.remove("d-none");
            }
        });
    }

    // Clear Form Button
    const btnClear = document.getElementById("btnClear");
    if (btnClear && reviewTextArea) {
        btnClear.addEventListener("click", function () {
            reviewTextArea.value = "";
            reviewTextArea.classList.remove("is-invalid");
            if (validationFeedback) validationFeedback.style.display = "none";
            charCounter.textContent = "0 / 2000 chars";
            charCounter.className = "badge bg-secondary-subtle text-secondary py-2 px-3 fw-medium";
        });
    }

    // ----------------------------------------------------
    // 3. Inject Review Samples (Home Page)
    // ----------------------------------------------------
    const btnGenuineSample = document.getElementById("btnGenuineSample");
    const btnFakeSample = document.getElementById("btnFakeSample");

    const sampleGenuine = "I purchased this coffee maker last week. The brewing speed is excellent. It has been running smoothly without any major issues. Overall, it is a decent value for money, though shipping took an extra day. I would recommend this to anyone looking for a reliable coffee maker.";
    const sampleFake = "!!! BEST WIRELESS HEADPHONES EVER !!! AMAZING QUALITY !!! BUY IT NOW !!! YOU WILL NOT REGRET IT !!! This smartwatch is a lifesaver! I bought 5 more for my family members. PERFECT!!!";

    if (btnGenuineSample && reviewTextArea) {
        btnGenuineSample.addEventListener("click", function () {
            reviewTextArea.value = sampleGenuine;
            reviewTextArea.dispatchEvent(new Event("input"));
            reviewTextArea.classList.remove("is-invalid");
            if (validationFeedback) validationFeedback.style.display = "none";
        });
    }

    if (btnFakeSample && reviewTextArea) {
        btnFakeSample.addEventListener("click", function () {
            reviewTextArea.value = sampleFake;
            reviewTextArea.dispatchEvent(new Event("input"));
            reviewTextArea.classList.remove("is-invalid");
            if (validationFeedback) validationFeedback.style.display = "none";
        });
    }

    // ----------------------------------------------------
    // 3b. Inject URL Samples & URL Validation (Home Page)
    // ----------------------------------------------------
    const productUrlInput = document.getElementById("productUrlInput");
    const productUrlForm = document.getElementById("productUrlForm");
    const urlValidationFeedback = document.getElementById("urlValidationFeedback");
    const btnAmazonSample = document.getElementById("btnAmazonSample");
    const btnFlipkartSample = document.getElementById("btnFlipkartSample");
    const btnClearUrl = document.getElementById("btnClearUrl");

    const sampleAmazon = "https://www.amazon.in/Apple-iPhone-15-Black-128GB/dp/B0CHX1W1Y2";
    const sampleFlipkart = "https://www.flipkart.com/apple-iphone-15-black-128-gb/p/itm2d82914";

    if (btnAmazonSample && productUrlInput) {
        btnAmazonSample.addEventListener("click", function () {
            productUrlInput.value = sampleAmazon;
            productUrlInput.classList.remove("is-invalid");
            if (urlValidationFeedback) urlValidationFeedback.style.display = "none";
        });
    }

    if (btnFlipkartSample && productUrlInput) {
        btnFlipkartSample.addEventListener("click", function () {
            productUrlInput.value = sampleFlipkart;
            productUrlInput.classList.remove("is-invalid");
            if (urlValidationFeedback) urlValidationFeedback.style.display = "none";
        });
    }

    if (btnClearUrl && productUrlInput) {
        btnClearUrl.addEventListener("click", function () {
            productUrlInput.value = "";
            productUrlInput.classList.remove("is-invalid");
            if (urlValidationFeedback) urlValidationFeedback.style.display = "none";
        });
    }

    if (productUrlForm && productUrlInput) {
        productUrlForm.addEventListener("submit", function (event) {
            const urlVal = productUrlInput.value.trim();
            if (!urlVal.startsWith("http://") && !urlVal.startsWith("https://")) {
                event.preventDefault();
                productUrlInput.classList.add("is-invalid");
                if (urlValidationFeedback) urlValidationFeedback.style.display = "block";
            } else {
                productUrlInput.classList.remove("is-invalid");
                if (urlValidationFeedback) urlValidationFeedback.style.display = "none";
                if (loadingOverlay) loadingOverlay.classList.remove("d-none");
            }
        });
    }

    // ----------------------------------------------------
    // 4. Chart.js Result Visuals (Result Page)
    // ----------------------------------------------------
    const resultChartCtx = document.getElementById("resultChart");
    const productDonutChartCtx = document.getElementById("productDonutChart");
    
    if (resultChartCtx && typeof probFake !== 'undefined' && typeof probGenuine !== 'undefined') {
        const labels = ['Genuine Prob', 'Fake Prob'];
        const data = [probGenuine, probFake];
        
        // Dynamic color theme dependent chart colors
        const textColor = getComputedStyle(document.documentElement).getPropertyValue('--text-primary').trim();
        
        new Chart(resultChartCtx, {
            type: 'doughnut',
            data: {
                labels: labels,
                datasets: [{
                    data: data,
                    backgroundColor: ['#22c55e', '#ef4444'],
                    borderColor: getComputedStyle(document.documentElement).getPropertyValue('--card-bg').trim() || '#ffffff',
                    borderWidth: 3
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                cutout: '70%',
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        callbacks: {
                            label: function (context) {
                                return `${context.label}: ${(context.raw * 100).toFixed(2)}%`;
                            }
                        }
                    }
                }
            }
        });
    }

    if (productDonutChartCtx && typeof probFake !== 'undefined' && typeof probGenuine !== 'undefined') {
        const labels = ['Genuine Reviews', 'Fake Reviews'];
        const data = [probGenuine, probFake];
        
        new Chart(productDonutChartCtx, {
            type: 'doughnut',
            data: {
                labels: labels,
                datasets: [{
                    data: data,
                    backgroundColor: ['#22c55e', '#ef4444'],
                    borderColor: getComputedStyle(document.documentElement).getPropertyValue('--card-bg').trim() || '#ffffff',
                    borderWidth: 3
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                cutout: '70%',
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        callbacks: {
                            label: function (context) {
                                return `${context.label}: ${context.raw}`;
                            }
                        }
                    }
                }
            }
        });
    }

    // ----------------------------------------------------
    // 5. Chart.js Dashboard Visuals (Dashboard Page)
    // ----------------------------------------------------
    const modelCompareChartCtx = document.getElementById("modelCompareChart");
    if (modelCompareChartCtx && typeof lrMetrics !== 'undefined' && typeof rfMetrics !== 'undefined') {
        const metricsLabels = ['Accuracy', 'Precision', 'Recall', 'F1 Score'];
        
        new Chart(modelCompareChartCtx, {
            type: 'bar',
            data: {
                labels: metricsLabels,
                datasets: [
                    {
                        label: 'Logistic Regression (Primary)',
                        data: [lrMetrics.accuracy, lrMetrics.precision, lrMetrics.recall, lrMetrics.f1_score],
                        backgroundColor: '#3b82f6',
                        borderRadius: 6
                    },
                    {
                        label: 'Random Forest (Comparison)',
                        data: [rfMetrics.accuracy, rfMetrics.precision, rfMetrics.recall, rfMetrics.f1_score],
                        backgroundColor: '#8b5cf6',
                        borderRadius: 6
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'top',
                        labels: {
                            color: getComputedStyle(document.documentElement).getPropertyValue('--text-secondary').trim()
                        }
                    }
                },
                scales: {
                    x: {
                        grid: { display: false },
                        ticks: { color: getComputedStyle(document.documentElement).getPropertyValue('--text-muted').trim() }
                    },
                    y: {
                        min: 0,
                        max: 1.0,
                        ticks: {
                            callback: function(value) { return (value * 100) + "%"; },
                            color: getComputedStyle(document.documentElement).getPropertyValue('--text-muted').trim()
                        },
                        grid: {
                            color: getComputedStyle(document.documentElement).getPropertyValue('--border-color').trim()
                        }
                    }
                }
            }
        });

        // ----------------------------------------------------
        // 6. Confusion Matrix Rendering (Dashboard Page)
        // ----------------------------------------------------
        // Matrix elements: TN, FP, FN, TP
        const cm = lrMetrics.confusion_matrix;
        const cellTN = document.getElementById("cellTN");
        const cellFP = document.getElementById("cellFP");
        const cellFN = document.getElementById("cellFN");
        const cellTP = document.getElementById("cellTP");

        if (cellTN && cellFP && cellFN && cellTP && cm) {
            cellTN.textContent = cm[0][0]; // TN
            cellFP.textContent = cm[0][1]; // FP
            cellFN.textContent = cm[1][0]; // FN
            cellTP.textContent = cm[1][1]; // TP
        }
    }

    // ----------------------------------------------------
    // 7. Clear History API Call (Dashboard Page)
    // ----------------------------------------------------
    const btnClearHistory = document.getElementById("btnClearHistory");
    if (btnClearHistory) {
        btnClearHistory.addEventListener("click", function () {
            if (confirm("Are you sure you want to delete all prediction history? This action cannot be undone.")) {
                fetch("/api/clear_history", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    }
                })
                .then(response => response.json())
                .then(data => {
                    if (data.status === "success") {
                        alert(data.message);
                        window.location.reload();
                    } else {
                        alert("Error clearing history: " + data.message);
                    }
                })
                .catch(err => {
                    console.error("Error:", err);
                    alert("A network error occurred.");
                });
            }
        });
    }

    // ----------------------------------------------------
    // 8. PDF Export and Printing (Result Page)
    // ----------------------------------------------------
    const btnPrintReport = document.getElementById("btnPrintReport");
    const btnDownloadPDF = document.getElementById("btnDownloadPDF");
    const pdfReportContent = document.getElementById("pdfReportContent");

    if (btnPrintReport) {
        btnPrintReport.addEventListener("click", function () {
            window.print();
        });
    }

    if (btnDownloadPDF && pdfReportContent) {
        btnDownloadPDF.addEventListener("click", function () {
            const activeTheme = htmlElement.getAttribute("data-theme");
            // Temporarily set light theme for clean PDF background rendering
            htmlElement.setAttribute("data-theme", "light");
            
            const opt = {
                margin:       [0.5, 0.5, 0.5, 0.5],
                filename:     `VeriTrust_Analysis_Report_#ID.pdf`,
                image:        { type: 'jpeg', quality: 0.98 },
                html2canvas:  { scale: 2, useCORS: true },
                jsPDF:        { unit: 'in', format: 'letter', orientation: 'portrait' }
            };

            // Custom dynamic filename if prediction ID is present
            const titleElem = document.querySelector(".pdf-header small");
            if (titleElem) {
                const text = titleElem.textContent;
                const match = text.match(/#\d+/);
                if (match) {
                    opt.filename = `VeriTrust_Analysis_Report_${match[0]}.pdf`;
                }
            }

            // Temporarily toggle elements to look better in PDF
            const pdfHeader = pdfReportContent.querySelector(".pdf-header");
            if (pdfHeader) pdfHeader.classList.remove("d-none");

            html2pdf().set(opt).from(pdfReportContent).save().then(() => {
                // Restore theme and hide PDF header
                htmlElement.setAttribute("data-theme", activeTheme);
                if (pdfHeader) pdfHeader.classList.add("d-none");
            });
        });
    }
});

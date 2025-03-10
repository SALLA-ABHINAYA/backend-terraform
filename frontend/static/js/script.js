       // Handle file upload
       function uploadFile() {
        let fileInput = document.getElementById('fileUpload').files[0];
        if (!fileInput) {
            alert("Please select a file first.");
            return;
        }
        
        let formData = new FormData();
        formData.append("file", fileInput);
        
        fetch('/upload', {
            method: 'POST',
            body: formData
        }).then(response => response.json())
        .then(data => alert(data.message))
        .catch(error => console.error('Error:', error));
    }

    // Fetch CSV file from FastAPI


    // Display CSV data in table
    function displayCsv(csvData) {
        let tableHead = document.getElementById('tableHead');
        let tableBody = document.getElementById('tableBody');
        tableHead.innerHTML = "";
        tableBody.innerHTML = "";

        let headers = csvData[0];
        headers.forEach(header => {
            let th = document.createElement("th");
            th.textContent = header;
            tableHead.appendChild(th);
        });

        csvData.slice(1).forEach(row => {
            let tr = document.createElement("tr");
            row.forEach(cell => {
                let td = document.createElement("td");
                td.textContent = cell;
                tr.appendChild(td);
            });
            tableBody.appendChild(tr);
        });
    }

    // Perform calculations
    function performCalculations() {
        fetch('/calculate')
        .then(response => response.json())
        .then(data => {
            console.log(data);
            document.getElementById('calculationResult').innerText = " Calculation Result: " + JSON.stringify(data, null, 2);
        })
        .catch(error => {
            document.getElementById('calculationResult').innerText = " Failed to perform calculations.";
            console.error('Error:', error);
        });
    }
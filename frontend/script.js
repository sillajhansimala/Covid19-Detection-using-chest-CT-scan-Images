document.getElementById('uploadForm').addEventListener('submit', function(e) {
    e.preventDefault(); // Prevent form submission

    const formData = new FormData();
    const imageFile = document.getElementById('fileInput').files[0];
    formData.append('file', imageFile); // Use 'file' instead of 'image'

    fetch('http://127.0.0.1:5000/predict', {
        method: 'POST',
        body: formData,
    })
    .then(response => response.json())
    .then(data => {
        if (data.predicted_class !== undefined) {
            document.getElementById('predictionResult').innerHTML = `Prediction: ${data.predicted_class}`;
        } else {
            document.getElementById('predictionResult').innerHTML = `Error: ${data.error}`;
        }
    })
    .catch(error => {
        document.getElementById('predictionResult').innerHTML = `Error: ${error.message}`;
    });
})
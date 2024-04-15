document.getElementById('translateButton').addEventListener('click', function() {
    var inputText = document.getElementById('inputText').value;
    fetch('/translate', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: inputText }),
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('outputText').value = data.translatedText;
    })
    .catch(error => {
        console.error('Error:', error);
    });
});

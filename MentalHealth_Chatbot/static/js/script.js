async function sendMessage() {
    let userInput = document.getElementById("userInput").value;
    if (!userInput.trim()) return;

    let chatbox = document.getElementById("chatbox");
    chatbox.innerHTML += "<div class='user-msg'>" + userInput + "</div>";

    const response = await fetch("/get", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ msg: userInput })
    });

    const data = await response.json();
    chatbox.innerHTML += "<div class='bot-msg'>" + data.response + "</div>";
    chatbox.scrollTop = chatbox.scrollHeight;
    document.getElementById("userInput").value = "";
}
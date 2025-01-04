document.addEventListener("DOMContentLoaded", () => {
  chrome.storage.local.get("currentPageURL", ({ currentPageURL }) => {
    if (currentPageURL) {
      fetch("http://localhost:8000/query", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ url: currentPageURL })
      })
        .then((response) => response.json())
        .then((data) => {
          console.log("Query sent successfully:", data);
        })
        .catch((error) => console.error("Error:", error));
    }
  });
});

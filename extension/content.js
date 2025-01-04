chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.type === "GET_PAGE_URL") {
      sendResponse({ url: window.location.href });
    }
  });
  
chrome.action.onClicked.addListener((tab) => {
  chrome.tabs.sendMessage(tab.id, { type: "GET_PAGE_URL" }, (response) => {
    if (response && response.url) {
      chrome.storage.local.set({ currentPageURL: response.url });
    }
  });
});

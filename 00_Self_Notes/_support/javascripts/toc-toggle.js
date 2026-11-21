/* Collapsible left table of contents.
   Adds an arrow button on the TOC panel's edge. The choice persists across
   pages (localStorage) and survives Material's instant navigation.
   Structure: layout.css. Colours: theme.css. */
(() => {
  const STORAGE_KEY = "atlas.toc.collapsed";
  const TOC_ID = "atlas-toc";
  const root = document.documentElement;
  const CHEVRON_LEFT =
    '<svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">' +
    '<path d="M15.41 7.41 14 6l-6 6 6 6 1.41-1.41L10.83 12z"/></svg>';

  const isCollapsed = () => root.dataset.atlasToc === "collapsed";

  function setCollapsed(collapsed) {
    if (collapsed) {
      root.dataset.atlasToc = "collapsed";
    } else {
      delete root.dataset.atlasToc;
    }
    try {
      localStorage.setItem(STORAGE_KEY, String(collapsed));
    } catch {
      /* Storage blocked (private mode): the toggle still works for this page. */
    }
  }

  function syncLabel(button) {
    const label = isCollapsed() ? "Show table of contents" : "Hide table of contents";
    button.setAttribute("aria-expanded", String(!isCollapsed()));
    button.setAttribute("aria-label", label);
    button.title = label;
  }

  function mount() {
    const sidebar = document.querySelector(".md-sidebar--secondary");
    if (!sidebar || sidebar.querySelector(".atlas-toc-toggle")) return;
    const toc = sidebar.querySelector(".md-sidebar__scrollwrap");
    if (toc) {
      toc.id = TOC_ID;
      // Material resizes the TOC as the page scrolls; mirror its height so CSS
      // can keep the button centred on the visible panel.
      const publishHeight = () =>
        sidebar.style.setProperty("--atlas-toc-panel-height", `${toc.offsetHeight}px`);
      new ResizeObserver(publishHeight).observe(toc);
    }

    const button = document.createElement("button");
    button.type = "button";
    button.className = "atlas-toc-toggle";
    button.setAttribute("aria-controls", TOC_ID);
    button.innerHTML = CHEVRON_LEFT;
    button.addEventListener("click", () => {
      setCollapsed(!isCollapsed());
      syncLabel(button);
    });
    sidebar.prepend(button);
    syncLabel(button);
  }

  try {
    if (localStorage.getItem(STORAGE_KEY) === "true") root.dataset.atlasToc = "collapsed";
  } catch {
    /* Storage blocked: start expanded. */
  }

  // Material re-emits document$ after every instant navigation.
  if (window.document$) {
    window.document$.subscribe(mount);
  } else {
    mount();
  }
})();

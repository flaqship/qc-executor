(function () {
  "use strict";

  var root = document.documentElement;

  function resolve(mode) {
    if (mode === "auto") {
      return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
    }
    return mode;
  }

  function applyMode(mode) {
    root.setAttribute("data-theme-mode", mode);
    root.setAttribute("data-theme", resolve(mode));
    localStorage.setItem("flaqship-theme", mode);
  }

  // Follow OS changes while in auto mode.
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", function () {
    if (root.getAttribute("data-theme-mode") === "auto") {
      root.setAttribute("data-theme", resolve("auto"));
    }
  });

  document.addEventListener("DOMContentLoaded", function () {
    // Theme toggle: cycles light -> dark -> auto.
    var toggle = document.getElementById("fq-theme-toggle");
    if (toggle) {
      toggle.addEventListener("click", function () {
        var order = ["light", "dark", "auto"];
        var current = root.getAttribute("data-theme-mode") || "auto";
        var next = order[(order.indexOf(current) + 1) % order.length];
        applyMode(next);
      });
    }

    // Mobile sidebar.
    var navToggle = document.getElementById("fq-nav-toggle");
    var sidebar = document.getElementById("fq-sidebar");
    var backdrop = document.getElementById("fq-sidebar-backdrop");
    function closeNav() {
      sidebar.classList.remove("fq-open");
      backdrop.classList.remove("fq-open");
      navToggle.setAttribute("aria-expanded", "false");
    }
    if (navToggle && sidebar && backdrop) {
      navToggle.addEventListener("click", function () {
        var open = sidebar.classList.toggle("fq-open");
        backdrop.classList.toggle("fq-open", open);
        navToggle.setAttribute("aria-expanded", String(open));
      });
      backdrop.addEventListener("click", closeNav);
    }

    // Version switcher: read a versions.json manifest and let the reader jump
    // to the same page in another version, falling back to that version's root.
    var vswitch = document.getElementById("fq-version-switcher");
    if (vswitch) {
      var vbtn = document.getElementById("fq-version-btn");
      var vmenu = document.getElementById("fq-version-menu");
      var vurl = vswitch.getAttribute("data-versions-url");
      var vcurrent = vswitch.getAttribute("data-current-version");
      var vpage = vswitch.getAttribute("data-pagename") || "index";
      var vopen = function () { vmenu.hidden = false; vbtn.setAttribute("aria-expanded", "true"); };
      var vclose = function () { vmenu.hidden = true; vbtn.setAttribute("aria-expanded", "false"); };
      vbtn.addEventListener("click", function (e) {
        e.stopPropagation();
        if (vmenu.hidden) { vopen(); } else { vclose(); }
      });
      document.addEventListener("click", function (e) { if (!vswitch.contains(e.target)) vclose(); });
      document.addEventListener("keydown", function (e) { if (e.key === "Escape") vclose(); });

      if (vurl) {
        var manifestAbs = new URL(vurl, window.location.href);
        fetch(vurl).then(function (r) { return r.json(); }).then(function (list) {
          if (!Array.isArray(list)) return;
          list.forEach(function (v) {
            var label = v.name || v.version || "";
            // Resolve the version's directory against the manifest location
            // (the site root), then the same page within that version.
            var versionRoot = new URL(v.url, manifestAbs);
            var target = new URL(vpage + ".html", versionRoot).href;
            var li = document.createElement("li");
            li.setAttribute("role", "option");
            var a = document.createElement("a");
            a.textContent = label;
            a.href = target;
            if (v.version === vcurrent || v.name === vcurrent) {
              li.className = "fq-version-active";
              a.setAttribute("aria-current", "true");
              var lbl = vswitch.querySelector(".fq-version-label");
              if (lbl) { lbl.textContent = label; }
            }
            a.addEventListener("click", function (ev) {
              ev.preventDefault();
              fetch(target, { method: "HEAD" }).then(function (res) {
                window.location.href = res.ok ? target : versionRoot.href;
              }).catch(function () { window.location.href = versionRoot.href; });
            });
            li.appendChild(a);
            vmenu.appendChild(li);
          });
        }).catch(function () { /* keep the static current-version label */ });
      }
    }

    // Collapsible left-nav sections: collapsed by default, with the branch
    // leading to the current page expanded so it stays oriented.
    var navItems = document.querySelectorAll(".fq-toctree li");
    Array.prototype.forEach.call(navItems, function (li) {
      var sub = li.querySelector(":scope > ul");
      if (!sub) return;
      li.classList.add("fq-has-children");
      var onCurrentPath = li.classList.contains("current");
      if (onCurrentPath) li.classList.add("fq-expanded");
      var btn = document.createElement("button");
      btn.type = "button";
      btn.className = "fq-toc-toggle";
      btn.setAttribute("aria-label", "Toggle subsection");
      btn.setAttribute("aria-expanded", String(onCurrentPath));
      btn.addEventListener("click", function (ev) {
        ev.preventDefault();
        ev.stopPropagation();
        var open = li.classList.toggle("fq-expanded");
        btn.setAttribute("aria-expanded", String(open));
      });
      li.insertBefore(btn, li.firstChild);
    });

    // Scroll-spy for the right-hand page TOC.
    var tocLinks = Array.prototype.slice.call(
      document.querySelectorAll(".fq-pagetoc a[href^='#']")
    );
    if (tocLinks.length && "IntersectionObserver" in window) {
      var map = {};
      tocLinks.forEach(function (link) {
        var id = decodeURIComponent(link.getAttribute("href").slice(1));
        var el = document.getElementById(id);
        if (el) map[id] = link;
      });
      var observer = new IntersectionObserver(function (entries) {
        entries.forEach(function (entry) {
          if (entry.isIntersecting) {
            tocLinks.forEach(function (l) { l.classList.remove("fq-active"); });
            var active = map[entry.target.id];
            if (active) active.classList.add("fq-active");
          }
        });
      }, { rootMargin: "-64px 0px -70% 0px" });
      Object.keys(map).forEach(function (id) {
        var el = document.getElementById(id);
        if (el) observer.observe(el);
      });
    }
  });
})();

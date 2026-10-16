/* O-SCD series — shared page behaviour. No dependencies. */
(function () {
  "use strict";

  var reduceMotion = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* ---------- theme toggle ---------- */
  function effectiveTheme() {
    var t = document.documentElement.getAttribute("data-theme");
    if (t) return t;
    return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }
  document.querySelectorAll(".theme-toggle").forEach(function (btn) {
    btn.addEventListener("click", function () {
      var next = effectiveTheme() === "dark" ? "light" : "dark";
      document.documentElement.setAttribute("data-theme", next);
      try { localStorage.setItem("oscd-theme", next); } catch (e) { /* storage unavailable */ }
      document.dispatchEvent(new CustomEvent("themechange"));
    });
  });

  /* ---------- videos ---------- */
  function loadVideo(v) {
    if (!v.getAttribute("src") && v.dataset.src) { v.src = v.dataset.src; v.load(); }
  }
  function playVideo(v) {
    if (!v || reduceMotion) return;
    loadVideo(v);
    var p = v.play();
    if (p && p.catch) p.catch(function () { /* autoplay blocked */ });
  }
  function syncCtrl(stage) {
    var btn = stage.querySelector(".video-ctrl");
    var v = activeVideo(stage);
    if (btn && v) btn.setAttribute("aria-pressed", v.paused ? "false" : "true");
  }
  function activeVideo(stage) {
    var panel = stage.querySelector('[role="tabpanel"]:not([hidden])');
    return (panel || stage).querySelector("video");
  }

  document.querySelectorAll("[data-video-stage]").forEach(function (stage) {
    var btn = stage.querySelector(".video-ctrl");
    stage.querySelectorAll("video").forEach(function (v) {
      v.addEventListener("play", function () { syncCtrl(stage); });
      v.addEventListener("pause", function () { syncCtrl(stage); });
    });
    if (btn) btn.addEventListener("click", function () {
      var v = activeVideo(stage);
      if (!v) return;
      if (v.paused) { loadVideo(v); v.play(); stage.dataset.userPaused = ""; }
      else { v.pause(); stage.dataset.userPaused = "1"; }
    });
    if ("IntersectionObserver" in window) {
      new IntersectionObserver(function (entries) {
        entries.forEach(function (e) {
          var v = activeVideo(stage);
          if (!v) return;
          if (e.isIntersecting) { if (!stage.dataset.userPaused) playVideo(v); }
          else v.pause();
        });
      }, { threshold: 0.25 }).observe(stage);
    } else {
      playVideo(activeVideo(stage));
    }
  });

  /* ---------- tabs ---------- */
  document.querySelectorAll("[data-tabs]").forEach(function (root) {
    var tabs = Array.prototype.slice.call(root.querySelectorAll('[role="tab"]'));
    function select(tab, focus) {
      tabs.forEach(function (t) {
        var on = t === tab;
        t.setAttribute("aria-selected", on ? "true" : "false");
        t.tabIndex = on ? 0 : -1;
        var panel = document.getElementById(t.getAttribute("aria-controls"));
        if (!panel) return;
        panel.hidden = !on;
        var v = panel.querySelector("video");
        if (v) {
          if (on) { v.currentTime = 0; if (!root.dataset.userPaused) playVideo(v); }
          else v.pause();
        }
      });
      if (focus) tab.focus();
      if (root.hasAttribute("data-video-stage")) syncCtrl(root);
    }
    tabs.forEach(function (t, i) {
      t.addEventListener("click", function () { select(t); });
      t.addEventListener("keydown", function (e) {
        var j = e.key === "ArrowRight" ? i + 1 : e.key === "ArrowLeft" ? i - 1 : e.key === "Home" ? 0 : e.key === "End" ? tabs.length - 1 : null;
        if (j === null) return;
        e.preventDefault();
        select(tabs[(j + tabs.length) % tabs.length], true);
      });
    });
  });

  /* ---------- before/after compare ---------- */
  document.querySelectorAll(".compare").forEach(function (c) {
    var input = c.querySelector('input[type="range"]');
    function set(v) { c.style.setProperty("--pos", v + "%"); }
    if (input) { input.addEventListener("input", function () { set(input.value); }); set(input.value); }
  });

  /* ---------- copy BibTeX ---------- */
  document.querySelectorAll("[data-copy]").forEach(function (btn) {
    var label = btn.querySelector("span");
    btn.addEventListener("click", function () {
      var src = document.querySelector(btn.getAttribute("data-copy"));
      if (!src) return;
      var text = src.textContent;
      function done() {
        if (!label) return;
        var old = label.textContent;
        label.textContent = "Copied";
        setTimeout(function () { label.textContent = old; }, 1600);
      }
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(done, fallback);
      } else fallback();
      function fallback() {
        var ta = document.createElement("textarea");
        ta.value = text; ta.style.position = "fixed"; ta.style.opacity = "0";
        document.body.appendChild(ta); ta.select();
        try { document.execCommand("copy"); done(); } catch (e) { /* ignore */ }
        document.body.removeChild(ta);
      }
    });
  });

  /* ---------- scatter / line chart ---------- */
  var NS = "http://www.w3.org/2000/svg";
  var COLORS = { oscd: "var(--oscd)", oscdpp: "var(--oscdpp)", neutral: "var(--neutral-mark)" };

  function el(name, attrs, parent) {
    var n = document.createElementNS(NS, name);
    for (var k in attrs) if (attrs[k] !== undefined) n.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(n);
    return n;
  }
  function starPath(cx, cy, r) {
    var d = "";
    for (var i = 0; i < 10; i++) {
      var rad = i % 2 ? r * 0.45 : r, a = -Math.PI / 2 + i * Math.PI / 5;
      d += (i ? "L" : "M") + (cx + rad * Math.cos(a)).toFixed(2) + " " + (cy + rad * Math.sin(a)).toFixed(2);
    }
    return d + "Z";
  }
  function crossPath(cx, cy, r) {
    return "M" + (cx - r) + " " + (cy - r) + "L" + (cx + r) + " " + (cy + r) + "M" + (cx + r) + " " + (cy - r) + "L" + (cx - r) + " " + (cy + r);
  }

  function renderChart(box) {
    var spec = JSON.parse(box.querySelector("script[type='application/json']").textContent);
    var old = box.querySelector("svg");
    if (old) old.remove();
    var W = Math.max(300, box.clientWidth);
    var narrow = W < 520;
    var H = Math.round(Math.min(460, Math.max(300, W * (narrow ? 0.85 : 0.56))));
    var m = { l: narrow ? 46 : 58, r: narrow ? 14 : 28, t: 18, b: 50 };
    var iw = W - m.l - m.r, ih = H - m.t - m.b;
    var X = function (v) { return m.l + (v - spec.x.min) / (spec.x.max - spec.x.min) * iw; };
    var Y = function (v) { return m.t + ih - (v - spec.y.min) / (spec.y.max - spec.y.min) * ih; };
    var fy = function (v) { return v.toFixed(spec.y.digits == null ? 2 : spec.y.digits); };

    var svg = el("svg", { viewBox: "0 0 " + W + " " + H, width: W, height: H, role: "img", "aria-label": spec.aria || "" });
    var grid = el("g", { "class": "grid" }, svg);
    spec.y.ticks.forEach(function (t) { el("line", { x1: m.l, x2: m.l + iw, y1: Y(t), y2: Y(t) }, grid); });
    var ax = el("g", { "class": "axis" }, svg);
    el("line", { x1: m.l, x2: m.l + iw, y1: m.t + ih, y2: m.t + ih }, ax);
    el("line", { x1: m.l, x2: m.l, y1: m.t, y2: m.t + ih }, ax);
    spec.x.ticks.forEach(function (t) {
      var g = el("g", { "class": "tick" }, svg);
      el("line", { x1: X(t), x2: X(t), y1: m.t + ih, y2: m.t + ih + 5, stroke: "var(--line-strong)" }, g);
      el("text", { x: X(t), y: m.t + ih + 20, "text-anchor": "middle" }, g).textContent = t;
    });
    spec.y.ticks.forEach(function (t) {
      var g = el("g", { "class": "tick" }, svg);
      el("text", { x: m.l - 8, y: Y(t) + 4, "text-anchor": "end" }, g).textContent = fy(t);
    });
    el("text", { "class": "axis-title", x: m.l + iw / 2, y: H - 8, "text-anchor": "middle" }, svg).textContent = spec.x.title;
    el("text", { "class": "axis-title", x: 14, y: m.t + ih / 2, "text-anchor": "middle", transform: "rotate(-90 14 " + (m.t + ih / 2) + ")" }, svg).textContent = spec.y.title;

    (spec.refs || []).forEach(function (r) {
      var c = COLORS[r.color] || r.color;
      el("line", { "class": "ref-line", x1: m.l, x2: m.l + iw, y1: Y(r.y), y2: Y(r.y), style: "stroke:" + c }, svg);
      if (r.label) el("text", { "class": "label muted", x: r.anchor === "end" ? m.l + iw - 4 : m.l + 6, y: Y(r.y) + (r.below ? 15 : -6), "text-anchor": r.anchor || "start" }, svg).textContent = r.label;
    });

    (spec.lines || []).forEach(function (ln) {
      var pts = spec.marks.filter(function (p) { return p.series === ln.series; });
      el("polyline", { "class": "series-line", points: pts.map(function (p) { return X(p.x) + "," + Y(p.y); }).join(" "), style: "stroke:" + COLORS[ln.color] }, svg);
    });

    var tip = box.querySelector(".chart-tip");
    if (!tip) { tip = document.createElement("div"); tip.className = "chart-tip"; box.appendChild(tip); }
    var hits = el("g", {}, null);

    spec.marks.forEach(function (p) {
      var cx = X(p.x), cy = Y(p.y), c = COLORS[p.color] || p.color, s = p.size || 6;
      if (p.shape === "star") el("path", { "class": "pt", d: starPath(cx, cy, s * 1.9), style: "fill:" + c, "stroke-width": 1.5 }, svg);
      else if (p.shape === "cross") el("path", { d: crossPath(cx, cy, s), style: "stroke:" + c + ";stroke-width:3;stroke-linecap:round;fill:none" }, svg);
      else if (p.shape === "square") el("rect", { "class": "pt", x: cx - s, y: cy - s, width: 2 * s, height: 2 * s, rx: 2, style: "fill:" + c }, svg);
      else el("circle", { "class": "pt", cx: cx, cy: cy, r: s, style: "fill:" + c }, svg);
      if (p.label && !(narrow && p.hideNarrow)) {
        var dx = p.dx == null ? 10 : p.dx, dy = p.dy == null ? 4 : p.dy;
        el("text", { "class": "label" + (p.strong ? " strong" : "") + (p.muted ? " muted" : ""), x: cx + dx, y: cy + dy, "text-anchor": p.anchor || "start" }, svg).textContent = p.label;
      }
      var h = el("circle", { "class": "hit", cx: cx, cy: cy, r: 16, tabindex: 0, "aria-label": p.name + ": " + (p.tip || "") }, hits);
      function show() {
        tip.innerHTML = "<b></b><span></span>";
        tip.firstChild.textContent = p.name;
        tip.lastChild.textContent = p.tip || "";
        var sx = box.clientWidth / W;
        tip.style.left = (cx * sx) + "px";
        tip.style.top = (cy * sx) + "px";
        tip.classList.add("on");
      }
      function hide() { tip.classList.remove("on"); }
      h.addEventListener("mouseenter", show); h.addEventListener("focus", show);
      h.addEventListener("mouseleave", hide); h.addEventListener("blur", hide);
    });
    svg.appendChild(hits);
    box.insertBefore(svg, box.firstChild);
  }

  document.querySelectorAll(".chart[data-chart]").forEach(function (box) {
    renderChart(box);
    if ("ResizeObserver" in window) {
      var last = box.clientWidth;
      new ResizeObserver(function () {
        if (Math.abs(box.clientWidth - last) > 4) { last = box.clientWidth; renderChart(box); }
      }).observe(box);
    }
  });
})();

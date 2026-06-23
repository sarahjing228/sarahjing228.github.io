// Site behavior lives here. Content itself is intentionally kept in index.html
// and local Markdown/HTML files so the website remains GitHub Pages friendly.
(function () {
  const root = document.documentElement;
  const themeButton = document.querySelector('#theme-toggle');
  const savedTheme = localStorage.getItem('sarah-site-theme');
  const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;

  function setTheme(theme) {
    root.dataset.theme = theme;
    themeButton.textContent = theme === 'dark' ? '☀' : '◐';
    themeButton.setAttribute('aria-label', `Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`);
  }
  setTheme(savedTheme || (prefersDark ? 'dark' : 'light'));
  themeButton.addEventListener('click', () => {
    const next = root.dataset.theme === 'dark' ? 'light' : 'dark';
    setTheme(next);
    localStorage.setItem('sarah-site-theme', next);
  });

  const header = document.querySelector('.site-header');
  const menuButton = document.querySelector('.menu-toggle');
  const nav = document.querySelector('#site-nav');
  window.addEventListener('scroll', () => header.classList.toggle('scrolled', window.scrollY > 10), { passive: true });
  menuButton.addEventListener('click', () => {
    const open = nav.classList.toggle('open');
    menuButton.setAttribute('aria-expanded', String(open));
  });
  nav.addEventListener('click', () => {
    nav.classList.remove('open');
    menuButton.setAttribute('aria-expanded', 'false');
  });

  document.querySelectorAll('.summary-column .show-more').forEach((button) => {
    button.addEventListener('click', () => {
      const column = button.closest('.summary-column');
      const expanded = column.classList.toggle('expanded');
      column.querySelectorAll('.summary-extra').forEach((item) => {
        item.hidden = !expanded;
      });
      button.textContent = expanded ? '− Show less' : '➕ Show more';
      button.setAttribute('aria-expanded', String(expanded));
    });
  });

  const experienceMap = document.querySelector('.experience-map-layout');
  if (experienceMap) {
    const activatePlace = (place) => {
      experienceMap.querySelectorAll('.map-location').forEach((item) => {
        const active = item.dataset.place === place;
        item.classList.toggle('active', active);
        item.setAttribute('aria-pressed', String(active));
      });
      experienceMap.querySelectorAll('.map-detail').forEach((detail) => {
        detail.classList.toggle('active', detail.dataset.place === place);
      });
    };

    experienceMap.addEventListener('click', (event) => {
      const locationButton = event.target.closest('.map-location');
      if (!locationButton) return;
      activatePlace(locationButton.dataset.place);
    });

    if (window.jsVectorMap && document.querySelector('#academic-map')) {
      const academicMarkers = [
        { name: 'Xi’an', coords: [34.3416, 108.9398], place: 'xian' },
        { name: 'Waterloo', coords: [43.4723, -80.5449], place: 'waterloo' },
        { name: 'Osaka', coords: [34.6937, 135.5023], place: 'osaka' },
        { name: 'NAIST M.S.', coords: [34.732, 135.735], place: 'nara-master' },
        { name: 'KIT', coords: [49.0069, 8.4037], place: 'kit' },
        { name: 'NAIST Ph.D.', coords: [34.704, 135.82], place: 'nara-phd' }
      ];
      new jsVectorMap({
        selector: '#academic-map',
        map: 'world',
        markers: academicMarkers,
        zoomButtons: false,
        zoomOnScroll: false,
        regionStyle: {
          initial: { fill: '#dbeafe', stroke: '#ffffff', strokeWidth: .6 },
          hover: { fill: '#bfdbfe' }
        },
        markerStyle: {
          initial: { fill: '#db2777', stroke: '#ffffff', strokeWidth: 3, r: 7 },
          hover: { fill: '#f472b6', stroke: '#ffffff', strokeWidth: 3, r: 9 }
        },
        labels: {
          markers: {
            render(marker) { return marker.name; }
          }
        },
        onMarkerClick(event, index) {
          activatePlace(academicMarkers[index].place);
        }
      });
    }
  }

  document.querySelector('#year').textContent = new Date().getFullYear();
  const revealObserver = new IntersectionObserver((entries) => {
    entries.forEach((entry) => entry.isIntersecting && entry.target.classList.add('visible'));
  }, { threshold: 0.08 });
  document.querySelectorAll('.reveal').forEach((element) => revealObserver.observe(element));

  // Lightweight local particle background: no CDN or third-party repository required.
  const canvas = document.querySelector('#particles');
  const context = canvas.getContext('2d');
  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  let particles = [];
  function resize() {
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = window.innerWidth * ratio;
    canvas.height = window.innerHeight * ratio;
    canvas.style.width = `${window.innerWidth}px`;
    canvas.style.height = `${window.innerHeight}px`;
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    const count = Math.min(80, Math.floor(window.innerWidth / 16));
    particles = Array.from({ length: count }, () => ({
      x: Math.random() * window.innerWidth,
      y: Math.random() * window.innerHeight,
      vx: (Math.random() - .5) * 5.8,
      vy: (Math.random() - .5) * 5.8,
      r: Math.random() * 2.2 + .7
    }));
  }
  function draw() {
    context.clearRect(0, 0, window.innerWidth, window.innerHeight);
    const dark = root.dataset.theme === 'dark';
    context.fillStyle = dark ? 'rgba(180,180,180,.45)' : 'rgba(136,136,136,.50)';
    context.strokeStyle = dark ? 'rgba(180,180,180,.18)' : 'rgba(136,136,136,.25)';
    particles.forEach((particle, index) => {
      if (!reduceMotion) {
        particle.x = (particle.x + particle.vx + window.innerWidth) % window.innerWidth;
        particle.y = (particle.y + particle.vy + window.innerHeight) % window.innerHeight;
      }
      context.beginPath(); context.arc(particle.x, particle.y, particle.r, 0, Math.PI * 2); context.fill();
      particles.slice(index + 1).forEach((other) => {
        const distance = Math.hypot(particle.x - other.x, particle.y - other.y);
        if (distance < 150) { context.beginPath(); context.moveTo(particle.x, particle.y); context.lineTo(other.x, other.y); context.stroke(); }
      });
    });
    if (!reduceMotion) requestAnimationFrame(draw);
  }
  resize(); draw();
  window.addEventListener('resize', resize);
}());

(() => {
  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const selectors = [
    '.section > h2',
    '.section > .prose',
    '.section > figure',
    '.stages > article',
    '.method-steps > article',
    '.resources > a',
    '#citation pre'
  ];
  const elements = [...document.querySelectorAll(selectors.join(','))];

  elements.forEach((element) => element.setAttribute('data-reveal', ''));
  document.documentElement.classList.add('reveal-ready');

  document.querySelectorAll('.stages, .method-steps, .resources').forEach((group) => {
    [...group.children].forEach((element, index) => {
      element.style.setProperty('--reveal-delay', `${index * 90}ms`);
    });
  });

  if (reduceMotion || !('IntersectionObserver' in window)) {
    elements.forEach((element) => element.classList.add('is-visible'));
    return;
  }

  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      entry.target.classList.toggle('is-visible', entry.isIntersecting);
    });
  }, {
    threshold: 0.08,
    rootMargin: '0px 0px -7% 0px'
  });

  elements.forEach((element) => observer.observe(element));
})();

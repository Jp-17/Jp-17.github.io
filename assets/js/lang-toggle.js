// Language toggle for bilingual blog posts
// Supports CN / EN switching with localStorage persistence

(function () {
  var STORAGE_KEY = 'blog-lang';
  var DEFAULT_LANG = 'cn';

  function applyLang(lang) {
    // Show/hide content sections
    document.querySelectorAll('.lang-cn, .lang-en').forEach(function (el) {
      el.classList.remove('active');
    });
    document.querySelectorAll('.lang-' + lang).forEach(function (el) {
      el.classList.add('active');
    });

    // Show correct title
    var cnTitle = document.querySelector('.post-title-cn');
    var enTitle = document.querySelector('.post-title-en');
    if (cnTitle) cnTitle.style.display = lang === 'cn' ? '' : 'none';
    if (enTitle) enTitle.style.display = lang === 'en' ? '' : 'none';

    // Update button states
    document.querySelectorAll('.lang-btn').forEach(function (btn) {
      btn.classList.toggle('active', btn.dataset.lang === lang);
    });
  }

  // Init on page load
  var saved = localStorage.getItem(STORAGE_KEY) || DEFAULT_LANG;
  applyLang(saved);

  // Global function called by onclick
  window.setLang = function (lang) {
    localStorage.setItem(STORAGE_KEY, lang);
    applyLang(lang);
  };
})();

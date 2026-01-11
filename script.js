function copyBibTeX() {
    var bibTexElement = document.querySelector(".bibtex-section pre code");
    var bibTexText = bibTexElement.innerText;
    navigator.clipboard.writeText(bibTexText);
    alert("BibTeX citation copied to clipboard!");
  }
  function toggleDarkMode() {
    document.body.classList.toggle("dark-mode");
    document.querySelector(".nav").classList.toggle("dark-mode");
  }
  window.onscroll = function () {
    const scrollUpBtn = document.getElementById("scrollUpBtn");
    if (
      document.body.scrollTop > 100 ||
      document.documentElement.scrollTop > 100
    ) {
      scrollUpBtn.style.display = "block";
    } else {
      scrollUpBtn.style.display = "none";
    }
  };
  function scrollToTop() {
    window.scrollTo({ top: 0, behavior: "smooth" });
  }
  class Carousel {
    constructor(element, interval = 3000) {
      this.container = element;
      this.track = element.querySelector(".carousel-track");
      this.slides = Array.from(element.querySelectorAll(".carousel-slide"));
      this.indicators = element.querySelector(".carousel-indicators");

      this.currentIndex = 0;
      this.slidesPerView = 3;
      this.totalSlides = Math.ceil(this.slides.length / this.slidesPerView);
      this.interval = interval;
      this.autoPlayTimer = null;

      this.createIndicators();
      this.setupEventListeners();
      this.startAutoPlay();
      this.updateCarousel();
    }

    createIndicators() {
      for (let i = 0; i < this.totalSlides; i++) {
        const button = document.createElement("button");
        button.classList.add("indicator");
        if (i === 0) button.classList.add("active");
        button.addEventListener("click", () => {
          this.goToSlide(i);
        });
        this.indicators.appendChild(button);
      }
    }

    setupEventListeners() {
      this.container
        .querySelector(".prev")
        .addEventListener("click", (e) => {
          e.preventDefault();
          this.prevSlide();
        });

      this.container
        .querySelector(".next")
        .addEventListener("click", (e) => {
          e.preventDefault();
          this.nextSlide();
        });

      this.container.addEventListener("mouseenter", () => {
        this.stopAutoPlay();
      });

      this.container.addEventListener("mouseleave", () => {
        this.startAutoPlay();
      });

      document.addEventListener("keydown", (e) => {
        if (this.container.matches(":hover")) {
          if (e.key === "ArrowLeft") {
            this.prevSlide();
          } else if (e.key === "ArrowRight") {
            this.nextSlide();
          }
        }
      });
    }

    updateCarousel() {
      const offset =
        -this.currentIndex *
        (100 / this.slidesPerView) *
        this.slidesPerView;
      this.track.style.transform = `translateX(${offset}%)`;

      const indicators = Array.from(this.indicators.children);
      indicators.forEach((indicator, index) => {
        indicator.classList.toggle("active", index === this.currentIndex);
      });
    }

    nextSlide() {
      this.currentIndex = (this.currentIndex + 1) % this.totalSlides;
      this.updateCarousel();
      this.resetAutoPlay();
    }

    prevSlide() {
      this.currentIndex =
        (this.currentIndex - 1 + this.totalSlides) % this.totalSlides;
      this.updateCarousel();
      this.resetAutoPlay();
    }

    goToSlide(index) {
      if (index !== this.currentIndex) {
        this.currentIndex = index;
        this.updateCarousel();
        this.resetAutoPlay();
      }
    }

    startAutoPlay() {
      if (this.autoPlayTimer) {
        clearInterval(this.autoPlayTimer);
      }
      this.autoPlayTimer = setInterval(() => {
        this.nextSlide();
      }, this.interval);
    }

    stopAutoPlay() {
      if (this.autoPlayTimer) {
        clearInterval(this.autoPlayTimer);
        this.autoPlayTimer = null;
      }
    }

    resetAutoPlay() {
      this.stopAutoPlay();
      this.startAutoPlay();
    }
  }

  document.addEventListener("DOMContentLoaded", () => {
    const imageCarousel = new Carousel(
      document.querySelector("#imageCarousel"),
      3000
    );
    const videoCarousel = new Carousel(
      document.querySelector("#videoCarousel"),
      5000
    );

    // Add touch support
    const carousels = [imageCarousel, videoCarousel];
    carousels.forEach((carousel) => {
      let touchStartX = 0;
      let touchEndX = 0;

      carousel.container.addEventListener(
        "touchstart",
        (e) => {
          touchStartX = e.changedTouches[0].screenX;
        },
        { passive: true }
      );

      carousel.container.addEventListener(
        "touchend",
        (e) => {
          touchEndX = e.changedTouches[0].screenX;
          handleSwipe(carousel);
        },
        { passive: true }
      );

      function handleSwipe(carousel) {
        const swipeThreshold = 50;
        const diff = touchStartX - touchEndX;

        if (Math.abs(diff) > swipeThreshold) {
          if (diff > 0) {
            carousel.nextSlide();
          } else {
            carousel.prevSlide();
          }
        }
      }
    });

    // Initialize lightbox for image carousel
    initializeLightbox();
  });

  // Lightbox functionality
  let currentLightboxIndex = 0;
  let lightboxImages = [];

  function initializeLightbox() {
    const imageCarousel = document.querySelector("#imageCarousel");
    if (!imageCarousel) return;

    const images = imageCarousel.querySelectorAll(".carousel-slide img");
    lightboxImages = Array.from(images).map(img => img.src);

    images.forEach((img, index) => {
      img.addEventListener("click", (e) => {
        e.stopPropagation();
        openLightbox(index);
      });
    });

    // Keyboard navigation for lightbox
    document.addEventListener("keydown", (e) => {
      const lightbox = document.getElementById("lightbox");
      if (!lightbox.classList.contains("active")) return;

      if (e.key === "Escape") {
        closeLightbox();
      } else if (e.key === "ArrowLeft") {
        navigateLightbox(-1);
      } else if (e.key === "ArrowRight") {
        navigateLightbox(1);
      }
    });
  }

  function openLightbox(index) {
    const lightbox = document.getElementById("lightbox");
    const lightboxImg = document.getElementById("lightbox-img");

    currentLightboxIndex = index;
    lightboxImg.src = lightboxImages[index];
    lightbox.classList.add("active");
    document.body.style.overflow = "hidden";
  }

  function closeLightbox() {
    const lightbox = document.getElementById("lightbox");
    lightbox.classList.remove("active");
    document.body.style.overflow = "";
  }

  function navigateLightbox(direction) {
    currentLightboxIndex = (currentLightboxIndex + direction + lightboxImages.length) % lightboxImages.length;
    const lightboxImg = document.getElementById("lightbox-img");
    lightboxImg.src = lightboxImages[currentLightboxIndex];
  }
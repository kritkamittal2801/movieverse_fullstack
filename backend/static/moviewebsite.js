// for sliding movie list wrapper
const arrows=document.querySelectorAll('.arrow');
const movielists=document.querySelectorAll('.movie-list');

// send activity to backend
async function postActivity(action, movie) {
    try {
        await fetch('/activity', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            credentials: 'same-origin',     // include cookies (session)
            body: JSON.stringify({ action, movie })
        });
    } catch (err) {
        console.warn('Activity post failed', err);
    }
}

function initSliders() {
document.querySelectorAll(".movie-list-wrapper").forEach((wrapper) => {
  const list = wrapper.querySelector(".movie-list");
  const rightArrow = wrapper.querySelector(".right-arrow");
  const leftArrow = wrapper.querySelector(".left-arrow");

  let counter = 0;
  const itemWidth = 290; // roughly one movie card width (adjust as needed)
  const visibleItems = Math.floor(window.innerWidth / 260);
  const totalItems = list.querySelectorAll(".movie-list-item").length;
  const maxCounter = totalItems - visibleItems;

  // Add smooth transition (important for visual effect)
  list.style.transition = "transform 0.8s ease-in-out";

  // Right arrow click
  if (rightArrow) {
    rightArrow.addEventListener("click", () => {
      if (counter < maxCounter) {
        counter++;
        list.style.transform = `translateX(-${counter * itemWidth}px)`;
      } else {
        // optional: loop back to start
        counter = 0;
        list.style.transform = `translateX(0)`;
      }
    });
  }

  // Left arrow click
  if (leftArrow) {
    leftArrow.addEventListener("click", () => {
      if (counter > 0) {
        counter--;
        list.style.transform = `translateX(-${counter * itemWidth}px)`;
      } else {
        // optional: jump to end if at start
        counter = maxCounter;
        list.style.transform = `translateX(-${counter * itemWidth}px)`;
      }
    });
  }
});}



// for changing dark theme
const ball=document.querySelector(".toggle-ball");
const container=document.querySelectorAll(".navbar,.navbar-container,.sidebar,.sidebar-item,.container,.toggle,.toggle-ball,.back-video,.featured-content,.faq-page,.faq-body,.faq-container,.faq-heading");

ball.addEventListener("click",()=>{
        container.forEach((item)=>{
            item.classList.toggle("active");
        })
});


// for expanding and contracting faq questions
const faq = document.getElementsByClassName("faq-page");
var i;
for (i = 0; i < faq.length; i++) {
    faq[i].addEventListener("click", function () {
        this.classList.toggle("plusminus");
        var body = this.nextElementSibling;
        if (body.style.display === "block") {
            body.style.display = "none";
        } 
        else {
            body.style.display = "block";
        }
    });
}


const searchIcon = document.getElementById('search-icon'); // sidebar icon
const searchModal = document.getElementById('search-modal');
const modalSearchInput = document.getElementById('modal-search');
const modalResultsDiv = document.getElementById('modal-results');

// Open modal on click
// Open modal on click — prevent the click from bubbling to window
searchIcon.addEventListener('click', (e) => {
    e.stopPropagation();                      // <- prevent the window click handler from running
    searchModal.style.display = 'block';
    document.body.classList.add('modal-active');
    modalSearchInput.focus();
});

// Close modal when clicking outside the modal
window.addEventListener('click', (e) => {
    // if modal is not open, ignore
    if (searchModal.style.display !== 'block') return;

    // if click happened inside modal, do nothing
    if (searchModal.contains(e.target)) return;

    // if click was on the search icon itself, do nothing (already handled)
    if (e.target === searchIcon) return;

    // otherwise close modal
    closeModal();
});

window.addEventListener('keydown', (e) => {
    if (e.key === "Escape") closeModal();
});

function closeModal() {
    searchModal.style.display = 'none';
    document.body.classList.remove('modal-active');
    modalSearchInput.value = '';
    modalResultsDiv.innerHTML = '';
}

function getTmdbUrl(title, tmdb_id) {
    if (tmdb_id && tmdb_id !== '' && !isNaN(tmdb_id)) {
        const cleanTitle = title
            .toLowerCase()
            .replace(/[\/\\]+/g, '-')     // replace any slashes
            .replace(/\s+/g, '-')          // replace spaces with dashes
            .replace(/[^a-z0-9\-]+/g, ''); // remove weird symbols
        return `https://www.themoviedb.org/movie/${tmdb_id}-${encodeURIComponent(cleanTitle)}`;
    } else {
        return `https://www.themoviedb.org/search?query=${encodeURIComponent(title)}`;
    }
}



let searchTimeout; // debounce timer
let loaderTimeout; // to delay showing the loader

modalSearchInput.addEventListener('input', () => {
    const query = modalSearchInput.value.trim();
    const loader = document.getElementById('search-loader');
    const resultsDiv = document.getElementById('modal-results');

    clearTimeout(searchTimeout);
    clearTimeout(loaderTimeout);

    if (!query) {
        resultsDiv.innerHTML = '';
        resultsDiv.style.display = 'none';
        loader.style.display = 'none';
        return;
    }

    // Wait 400ms after user stops typing before making request
    searchTimeout = setTimeout(async () => {
        // Delay loader display — only show if it’s taking time
        loaderTimeout = setTimeout(() => {
            loader.style.display = 'flex';
            resultsDiv.style.display = 'none';
        }, 200); // loader appears only if slow

        try {
            const response = await fetch(`/api/search?q=${encodeURIComponent(query)}`);
            const results = await response.json();

            // Stop loader
            clearTimeout(loaderTimeout);
            loader.style.display = 'none';
            resultsDiv.style.display = 'block';
            resultsDiv.classList.remove('show');
            resultsDiv.offsetHeight; // force reflow for animation reset

            if (results.length === 0) {
                resultsDiv.innerHTML = '<p>No movies found</p>';
            } else {
                resultsDiv.innerHTML = results.map(movie => `
                    <div class="search-result"
                         data-title="${movie.title}"
                         data-tmdb-id="${movie.tmdb_id}">
                        <img src="${movie.image || 'static/img/default.jpg'}" class="search-result-img" alt="${movie.title}">
                        <div class="search-result-text">
                            <b>${movie.title}</b><br>
                            <small>${movie.description}</small>
                        </div>
                    </div>
                `).join('');

                document.querySelectorAll('.search-result').forEach(el => {
                    el.addEventListener('click', () => {
                        const title = el.dataset.title;
                        const tmdbId = el.dataset.tmdbId;
                        if (!title || title === 'undefined') return;

                        postActivity('click', title).then(() => {
                            window.open(getTmdbUrl(title, tmdbId), '_blank');
                        });
                    });
                });
            }

            // Smooth fade-in animation
            requestAnimationFrame(() => resultsDiv.classList.add('show'));
        } catch (err) {
            clearTimeout(loaderTimeout);
            loader.style.display = 'none';
            resultsDiv.style.display = 'block';
            resultsDiv.innerHTML = '<p>Error fetching results</p>';
            console.error(err);
        }
    }, 400); // wait 400ms after typing stops
});

/* ---------- Log searches on Enter ---------- */
modalSearchInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
        const query = modalSearchInput.value.trim();
        if (query) {
            postActivity('search', query);
        }
    }
});


async function loadRecommendations() {
    try {
        const response = await fetch('/recommendations');
        const data = await response.json();

        const recList = document.getElementById('recommended-for-you-list');
        recList.innerHTML = '';
        data.recommended_for_you.forEach(movie => {
            const div = document.createElement('div');
            div.className = 'movie-list-item';
            div.innerHTML = `
                <img class="movie-list-item-img" src="${movie.image}" alt="${movie.title}">
                <span class="movie-list-item-title">${movie.title}</span>
                <p class="movie-list-item-desc">${movie.description}</p>
                <button class="movie-list-item-button">WATCH</button>
            `;
                div.onclick = () => {
    postActivity('click', movie.title).then(() => {
            window.open(getTmdbUrl(movie.title, movie.tmdb_id), '_blank');
        });
};


            recList.appendChild(div);
        });

       // Because you watched section
const watchedList = document.getElementById('because-you-watched-list');
const watchedSection = document.getElementById('because-you-watched-section');

if (data.because_you_watched && data.because_you_watched.recommendations.length > 0) {
    watchedSection.style.display = 'block';

    // Use the last_watched field for heading
    document.getElementById('last-watched-title').textContent = data.because_you_watched.last_watched;

    watchedList.innerHTML = '';
    data.because_you_watched.recommendations.forEach(movie => {
        const div = document.createElement('div');
        div.className = 'movie-list-item';
        div.innerHTML = `
            <img class="movie-list-item-img" src="${movie.image}" alt="${movie.title}">
            <span class="movie-list-item-title">${movie.title}</span>
            <p class="movie-list-item-desc">${movie.description}</p>
            <button class="movie-list-item-button">WATCH</button>
        `;
        div.onclick = () => {
   postActivity('click', movie.title).then(() => {
            window.open(getTmdbUrl(movie.title, movie.tmdb_id), '_blank');
        });
};


        watchedList.appendChild(div);
    });
} else {
    watchedSection.style.display = 'none';
}


    } catch (err) {
        console.error('Failed to load recommendations:', err);
    }
    initSliders();
}

document.addEventListener('DOMContentLoaded', () => {
    loadRecommendations();
});

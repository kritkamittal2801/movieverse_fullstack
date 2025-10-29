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


arrows.forEach((arrow,ind)=>{
    let counter=0;
    arrow.addEventListener("click",()=>{
        const ratio=Math.floor(window.innerWidth/260);
        let itemnumber=movielists[ind].querySelectorAll('img').length;
        if(counter<(itemnumber-5+5-ratio)){
            movielists[ind].style.transform= `translateX(${movielists[ind].computedStyleMap().get("transform")[0].x.value-290}px)`;
            counter++;
        }
        else{
            movielists[ind].style.transform= `translateX(0)`;
            counter=0;
        }
    })
});

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




// Live search
modalSearchInput.addEventListener('input', async () => {
    const query = modalSearchInput.value.trim();
    if (!query) {
        modalResultsDiv.innerHTML = '';
        return;
    }

    try {
        const response = await fetch(`/api/search?q=${encodeURIComponent(query)}`);
        const results = await response.json();

        if (results.length === 0) {
            modalResultsDiv.innerHTML = '<p>No movies found</p>';
        } else {
            modalResultsDiv.innerHTML = results.map(movie => `
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

// Add click handlers to each result
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
    } catch (err) {
        modalResultsDiv.innerHTML = '<p>Error fetching results</p>';
        console.error(err);
    }
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
}

document.addEventListener('DOMContentLoaded', () => {
    loadRecommendations();
});

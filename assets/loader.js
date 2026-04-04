const SLIDES = [
    '01_title.html', '02_motivation.html', '03_real_data.html', '04_standard_nmf.html',
    '05_formulation.html', '06_challenges.html', '07_algo_mu.html', '08_algo_am.html',
    '09_baselines.html', '10_convergence_mu.html', '10_convergence_am.html', '11_datasets.html',
    '12_setup.html', '13_max_error.html', '14_disparity.html', '15_tradeoff.html',
    '16_cross_comparison.html', '17_when_it_works.html', '18_robustness.html', '19_pareto.html',
    '20_downstream.html', '21_latent_space.html', '22_conclusions.html', '23_future_work.html', '24_final.html'
];

class SlideEngine {
    constructor() {
        this.currentIndex = 0;
        this.container = document.getElementById('slide-content');
        this.totalSlides = SLIDES.length;
        
        this.init();
    }

    async init() {
        // Init total count
        document.getElementById('total-slides').textContent = this.totalSlides.toString().padStart(2, '0');
        
        // Navigation events
        document.getElementById('prev-btn').addEventListener('click', () => this.prev());
        document.getElementById('next-btn').addEventListener('click', () => this.next());
        
        window.addEventListener('keydown', (e) => {
            if (e.key === 'ArrowRight' || e.key === ' ') this.next();
            if (e.key === 'ArrowLeft') this.prev();
        });

        // Hash change for deep linking
        window.addEventListener('hashchange', () => this.loadFromHash());
        
        this.loadFromHash();
    }

    loadFromHash() {
        const hash = window.location.hash.replace('#slide-', '');
        const targetIndex = parseInt(hash) - 1;
        
        if (!isNaN(targetIndex) && targetIndex >= 0 && targetIndex < this.totalSlides) {
            this.goTo(targetIndex);
        } else {
            this.goTo(0);
        }
    }

    async goTo(index) {
        if (index === this.currentIndex && this.container.innerHTML !== '') return;
        
        // Transition out
        this.container.classList.add('slide-exit');
        
        setTimeout(async () => {
            try {
                const response = await fetch(`slides/${SLIDES[index]}`);
                if (!response.ok) throw new Error('Slide not found');
                
                const html = await response.text();
                this.currentIndex = index;
                this.updateUI();
                
                this.container.innerHTML = html;
                window.location.hash = `slide-${index + 1}`;
                
                // Transition in
                this.container.classList.remove('slide-exit');
                this.container.classList.add('slide-enter');
                
                setTimeout(() => {
                    this.container.classList.remove('slide-enter');
                }, 50);

            } catch (err) {
                console.error('Error loading slide:', err);
                this.container.innerHTML = `
                    <div class="slide center">
                        <h2 style="color: var(--accent-acc)">ERRO DE CARREGARE [0x404]</h2>
                        <p>Slide not found: slides/${SLIDES[index]}</p>
                    </div>
                `;
            }
        }, 300);
    }

    next() {
        if (this.currentIndex < this.totalSlides - 1) {
            this.goTo(this.currentIndex + 1);
        }
    }

    prev() {
        if (this.currentIndex > 0) {
            this.goTo(this.currentIndex - 1);
        }
    }

    updateUI() {
        document.getElementById('current-slide').textContent = (this.currentIndex + 1).toString().padStart(2, '0');
        const progress = ((this.currentIndex + 1) / this.totalSlides) * 100;
        document.getElementById('progress-bar').style.width = `${progress}%`;
    }
}

// Start the engine
window.addEventListener('DOMContentLoaded', () => {
    window.presentation = new SlideEngine();
});

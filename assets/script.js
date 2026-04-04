/**
 * Executive Presentation Script for Fairer-NMF
 * Includes Reveal.js init and KaTeX math rendering
 */

document.addEventListener('DOMContentLoaded', () => {
    // 1. Initialize Reveal.js
    Reveal.initialize({
        hash: true,
        slideNumber: 'c/t',
        progress: true,
        controls: true,
        center: false,
        width: 1280,
        height: 720,
        margin: 0.1,
        minScale: 0.2,
        maxScale: 2.0,
        transition: 'slide', // 'none' | 'fade' | 'slide' | 'convex' | 'concave' | 'zoom'
        transitionSpeed: 'default',
        backgroundTransition: 'fade',
        plugins: [] // We'll handle extensions manually for better control
    });

    // 2. Initialize KaTeX Auto-Render
    // This allows $...$ and $$...$$ to be rendered as math
    if (window.renderMathInElement) {
        renderMathInElement(document.body, {
            delimiters: [
                {left: '$$', right: '$$', display: true},
                {left: '$', right: '$', display: false},
                {left: '\\(', right: '\\)', display: false},
                {left: '\\[', right: '\\]', display: true}
            ],
            throwOnError: false
        });
    }

    // 3. Optional: Subtle background parallax or animation
    // Can be added here for the "WOW" factor
    console.log("Presentation Engine Initialized: Executive Glassmorphism Mode");
});

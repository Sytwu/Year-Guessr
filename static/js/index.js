window.HELP_IMPROVE_VIDEOJS = false;


$(document).ready(function() {
    // Check for click events on the navbar burger icon

    // Comment out carousel initialization since we're using tabs now
    /*
    var options = {
		slidesToScroll: 1,
		slidesToShow: 1,
		loop: true,
		infinite: true,
		autoplay: true,
		autoplaySpeed: 5000,
    }

	// Initialize all div with carousel class
    var carousels = bulmaCarousel.attach('.carousel', options);
    */
	
    bulmaSlider.attach();

})

// Tab switching function
function openTab(evt, tabId) {
    // Get the parent container to scope the search
    var parentContainer = evt.currentTarget.closest('.carousel-container');
    
    // Hide all tab panes in this container
    var tabPanes = parentContainer.querySelectorAll('.tab-pane');
    for (var i = 0; i < tabPanes.length; i++) {
        tabPanes[i].classList.remove('active');
    }
    
    // Remove active class from all tab buttons in this container
    var tabButtons = parentContainer.querySelectorAll('.tab-btn');
    for (var i = 0; i < tabButtons.length; i++) {
        tabButtons[i].classList.remove('active');
    }
    
    // Show the selected tab pane
    document.getElementById(tabId).classList.add('active');
    
    // Add active class to the clicked button
    evt.currentTarget.classList.add('active');
}

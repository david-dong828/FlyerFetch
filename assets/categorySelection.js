
document.querySelectorAll(".toggle-button").forEach(btn => {
    btn.addEventListener("click", function() {
        this.classList.toggle("btn-primary");
        this.classList.toggle("btn-outline-primary");
    });
});


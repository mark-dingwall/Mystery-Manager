/* Toggle unchanged items visibility */
function toggleUnchanged(el) {
    var items = el.nextElementSibling;
    if (items.style.display === "none") {
        items.style.display = "block";
        el.classList.add("expanded");
    } else {
        items.style.display = "none";
        el.classList.remove("expanded");
    }
}

/* Sort box cards */
document.addEventListener("DOMContentLoaded", function () {
    var buttons = document.querySelectorAll(".btn-sort");
    var container = document.getElementById("box-pairs");
    if (!container) return;

    buttons.forEach(function (btn) {
        btn.addEventListener("click", function () {
            buttons.forEach(function (b) { b.classList.remove("active"); });
            btn.classList.add("active");

            var sortKey = btn.getAttribute("data-sort");
            var cards = Array.from(container.querySelectorAll(".box-card"));

            cards.sort(function (a, b) {
                if (sortKey === "name") {
                    return a.getAttribute("data-name").localeCompare(b.getAttribute("data-name"));
                }
                if (sortKey === "delta") {
                    return parseFloat(b.getAttribute("data-delta")) - parseFloat(a.getAttribute("data-delta"));
                }
                if (sortKey === "manual-score") {
                    return parseFloat(a.getAttribute("data-manual-score")) - parseFloat(b.getAttribute("data-manual-score"));
                }
                if (sortKey === "algo-score") {
                    return parseFloat(a.getAttribute("data-algo-score")) - parseFloat(b.getAttribute("data-algo-score"));
                }
                return 0;
            });

            cards.forEach(function (card) { container.appendChild(card); });
        });
    });
});

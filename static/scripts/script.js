document.addEventListener('DOMContentLoaded', function() {
  var price_movement_cells = document.getElementById("price_movement");
  var price_cells = document.getElementById("price");
  if (price_movement_cells != null) {
    if (parseFloat(price_movement_cells.innerHTML, 10) < 0) {
      price_cells.style.backgroundColor = "red";
      price_movement_cells.insertAdjacentHTML('afterbegin', '⇩ ');
    }
    else if (parseFloat(price_movement_cells.innerHTML, 10) > 0) {
      price_cells.style.backgroundColor = "green";
      price_movement_cells.insertAdjacentHTML('afterbegin', '⇧ ');
    }
  }

  var sentiment_cells = document.getElementsByClassName("sentiment");
  if (sentiment_cells != null) {
    for (var i = 0; i < sentiment_cells.length; i++) {
      if (sentiment_cells[i].innerHTML === "negative") {
        sentiment_cells[i].style.backgroundColor = "red";
      }
      else if (sentiment_cells[i].innerHTML === "positive") {
        sentiment_cells[i].style.backgroundColor = "green";
      }
    }
  }
}, false);

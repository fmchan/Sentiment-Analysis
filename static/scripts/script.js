document.addEventListener('DOMContentLoaded', function() {
  var price_movement_cells = document.getElementsByClassName("price_movement");
  var price_cells = document.getElementsByClassName("price");
  var sentiment_cells = document.getElementsByClassName("sentiment");

  if (parseFloat(price_movement_cells[0].innerHTML, 10) < 0) {
    price_cells[0].style.backgroundColor = "red";
    price_movement_cells[0].insertAdjacentHTML('afterbegin', '⇩ ');
  }
  else if (parseFloat(price_movement_cells[0].innerHTML, 10) > 0) {
    price_cells[0].style.backgroundColor = "green";
    price_movement_cells[0].insertAdjacentHTML('afterbegin', '⇧ ');
  }

  for (var i = 0; i < sentiment_cells.length; i++) {
      if (sentiment_cells[i].innerHTML === "negative") {
        sentiment_cells[i].style.backgroundColor = "red";
      }
      else if (sentiment_cells[i].innerHTML === "positive") {
        sentiment_cells[i].style.backgroundColor = "green";
      }
  }
}, false);

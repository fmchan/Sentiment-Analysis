document.addEventListener('DOMContentLoaded', function() {
  var cells = document.getElementsByClassName("sentiment");
  for (var i = 0; i < cells.length; i++) {
      if (cells[i].innerHTML === "negative") {
          cells[i].style.backgroundColor = "red";
      }
      else if (cells[i].innerHTML === "positive") {
        cells[i].style.backgroundColor = "green";
      }
  }
}, false);

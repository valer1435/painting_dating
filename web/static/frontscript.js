var GLOBAL_FLAG = 0


target_age  = [" grim darkness", " Church and the Thy word", " Memmi and Esded", " Konrad Witz", " Bosch", " Michelangelo", " Vasari", " Aert",
" Pussino", " Fradel", " Coccorante", " Issel" ," Aivazovsky"]

age  = ["1300 and older", "1300-1350", "1351-1400", "1401-1450", "1451-1500", "1501-1550", "1551-1600", "1601-1650", "1651-1700", "1701-1750",
  "1751-1800", "1801-1850", "1851-1900"]

window.onload = function() {



$(".zone").fadeOut(1)

let dropArea = document.getElementById('dropzone')

;['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
  dropArea.addEventListener(eventName, preventDefaults, false)
})

function preventDefaults (e) {
  e.preventDefault()
  e.stopPropagation()
}

;['dragover'].forEach(eventName => {
  dropArea.addEventListener(eventName, highlight, false)
})
;['drop'].forEach(eventName => {
  dropArea.addEventListener(eventName, unhighlight, false)
})
function highlight(e) {
  $(".zone").fadeIn(300)
}
function unhighlight(e) {
 $(".zone").fadeOut(300)
}

function uploadFile(file) {
  $(".req").fadeOut(300)
  let url = '/upload'
  let formData = new FormData()
  formData.append('file', file)
  fetch(url, {
    method: 'POST',
    body: formData
  })
  .then((response) => {
      return response.json()
  }).then((data) => {
    console.log(data.age);
    GLOBAL_FLAG = 1;
    document.getElementsByClassName("req")[0].innerHTML = "<span>" + age[data.age].toString() + " — the age of" + target_age[data.age].toString() + ". Try again? </span> ";
    $(".req").fadeIn(600);
  })
  .catch(() => {})
}

function handleFiles(files) {
  ([...files]).forEach(uploadFile)
}

dropArea.addEventListener('drop', handleDrop, false)
function handleDrop(e) {
  let dt = e.dataTransfer
  let files = dt.files
  handleFiles(files)
}


};
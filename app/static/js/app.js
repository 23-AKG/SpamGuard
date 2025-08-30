const dz = document.getElementById('dropzone');
const fi = document.getElementById('fileInput');
if (dz && fi){
  ['dragenter','dragover'].forEach(ev=>dz.addEventListener(ev,e=>{e.preventDefault();dz.classList.add('drag')}));
  ['dragleave','drop'].forEach(ev=>dz.addEventListener(ev,e=>{e.preventDefault();dz.classList.remove('drag')}));
  dz.addEventListener('drop', e => {
    const files = e.dataTransfer.files;
    fi.files = files;
  });
}

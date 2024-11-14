const nama = "nua";
let usia = 20;

const biodata = document.getElementById('biodata');
console.log(biodata);

function generateBiodata() {
    let generasi; 

    if(usia > 10 && usia < 18) { 
        generasi = "Generasi Remaja";
    } 
    else if (usia > 18 && usia < 30) {
        generasi = "Generasi Dewasa";
    } 
    else if (usia >= 30){
        generasi = "Generasi Tua ";
    
    }
    else if (usia > 2 && usia < 10) {
        generasi = "Generasi Anak-anak";
    }
    else {
        generasi = "Generasi Bayi";
    } 

    return biodata.innerHTML = generasi;
}

console.log(nama);
console.log(usia);

generateBiodata();
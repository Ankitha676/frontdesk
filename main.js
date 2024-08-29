let Ball = document.getElementsByClassName("ball");
document.onmousemove = function() {

  //get innerWidth,innerheight for browser
  
    //get horizontal coordinate of the onmousemove
    let x = event.clientX * 100 / window.innerWidth + "%";

    //get the verticle coordinate of onmousemove
    let y = event.clientY * 100 / window.innerHeight + "%";

    for(let i=0;i<2;i++) {
        Ball[i].style.left = x;
        Ball[i].style.top = y;
        Ball[i].style.transform = "translate(-"+x+", -"+y+")";
    }
  
}
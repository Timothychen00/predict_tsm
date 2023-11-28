function get_data(){
    fetch('http://127.0.0.1:5600/get_api',{mode:'cors'})
    .then((res)=>(res.json()))
    .then((res)=>{
        render_chart(res);
        console.log(res);
    })
    
}
get_data()
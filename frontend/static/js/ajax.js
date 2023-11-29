function get_data(){
    fetch('http://127.0.0.1:5600/get_api',{mode:'cors'})
    .then((res)=>(res.json()))
    .then((res)=>{
        // console.log(res);
        res=res['data']
        res[0]=res[0]
        res[1]=res[1]
        res[2]=res[2]
        console.log(res[2])
        // 
        render_chart(res);
        
    })
    
}
get_data()
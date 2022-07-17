$(document).ready(function(){
var log_url=$("#execution_id").val();
if (log_url==="None")
{
log_url="";
}
$("#log-panel").hide();
var is_visible_log=false;

$("#close-log-menu").click(function(){
$("#log-panel").hide();
$("#generate_log").html("View log<i class='fas fa-angle-down fa-1x'></i>");
});
$("#generate_log").click(function(){
is_visible_log=!is_visible_log;
$("#log-panel").slideToggle();
if (is_visible_log)
{
$("#generate_log").html("Close log<i class='fas fa-angle-up fa-1x'></i>");
 event.preventDefault();
 event.stopImmediatePropagation();

var existing_src=$('#log_load').attr('src');
if (log_url!="")
{
 var url = '/stream?project_id='+$("#project_id").val()+'&execution_id='+log_url;
 if (existing_src!=url)
 {
 $('#log_load').attr('src', url);
 }
 }
}
else
{
$("#generate_log").html("View log<i class='fas fa-angle-down fa-1x'></i>");
}
})
    $("#train_model").click(function(){

    var project_id=$("#project_id").val();
    var sentiment_project_id="";
    var sentiment_user_id="";
    var sentiment_data="";
    if (project_id=='16')
    {
    sentiment_project_id = prompt("Projet Id", "");

    sentiment_user_id=prompt("User Id", "");
    sentiment_data=prompt("Data", "");

    if (sentiment_project_id=="" || sentiment_user_id=="" || sentiment_data=="")
    {
    alert("Sentiment project requires data to begin training");
    return false;
    }

    if (sentiment_project_id==null || sentiment_user_id==null || sentiment_data==null)
    {
    alert("Sentiment project requires data to begin training");
    return false;
    }
    }
     $("#train_model").prop('disabled', true);
      $("#msg-status").html("<div class='col-md-1 col-md-offset-5'><i style='color:#008cba;' class='fas fa-sync fa-spin fa-1x'></i></div>");
    $.ajax({
                     url:'/train',
                     type: 'POST',
                     dataType: 'json',
                     data:  JSON.stringify({
                                        project_id:project_id,
                                        sentiment_project_id:sentiment_project_id,
                                        sentiment_user_id:sentiment_user_id,
                                        sentiment_data:sentiment_data



                                    }),
                     contentType: 'application/json',

                     success: function(response)
                     {
                         if (response['status'])
                             {
                            $('#execution_id').val(response['execution_id']);
                            log_url=response['execution_id'];
                            $("#msg-status").html("<div class='alert alert-success' role='alert'> "+response['message']+"<a class='close'  onclick=\"$('.alert').fadeOut(1000)\">&times;</a></div>")
                             $("#train_model").prop('disabled', false);

                             }
                          else
                              {
                               log_url=response['execution_id'];
                              $("#msg-status").html("<div class='alert alert-info' role='alert'> "+response['message']+"<a class='close'  onclick=\"$('.alert').fadeOut(1000)\">&times;</a></div>")
                              $("#train_model").prop('disabled', false);

                              }
                     },
                     error: function(error)
                     {
                         log_url=response['execution_id'];
                         $("#msg-status").html("<div class='alert alert-danger' role='alert'> "+response['message']+"<a class='close'  onclick=\"$('.alert').fadeOut(1000)\">&times;</a></div>")
                         $("#train_model").prop('disabled', false);

                     }






        });
    });
    $("#predict_from_model").click(function(){

         var project_id=$("#project_id").val();
         var sentiment_project_id="";
    var sentiment_user_id="";
    var sentiment_data="";
    if (project_id=='16')
    {
    sentiment_project_id = prompt("Projet Id", "");

    sentiment_user_id=prompt("User Id", "");
    sentiment_data=prompt("text", "");

    if (sentiment_project_id=="" || sentiment_user_id=="" || sentiment_data=="")
    {
    alert("Sentiment project requires data to begin training");
    return false;
    }

    if (sentiment_project_id==null || sentiment_user_id==null || sentiment_data==null)
    {
    alert("Sentiment project requires data to begin training");
    return false;
    }
    }
          $("#predict_from_model").prop('disabled', true);
           $("#msg-status").html("<div class='col-md-1 col-md-offset-5'><i style='color:#008cba;' class='fas fa-sync fa-spin fa-1x'></i></div>");
        $.ajax({
                     url:'/predict',
                     type: 'POST',
                     dataType: 'json',
                     data:  JSON.stringify({
                                        project_id:project_id,
                                        sentiment_project_id:sentiment_project_id,
                                        sentiment_user_id:sentiment_user_id,
                                        sentiment_data:sentiment_data

                                    }),
                     contentType: 'application/json',

                     success: function(response)
                     {

                         if (response['status'])
                             {
 $('#execution_id').val(response['execution_id']);
                            $("#msg-status").html("<div class='alert alert-success' role='alert'> "+response['message']+"<a class='close'  onclick=\"$('.alert').fadeOut(1000)\">&times;</a></div>")
                                  $("#predict_from_model").prop('disabled', false);
                                  }
                          else
                              {
                               $('#execution_id').val(response['execution_id']);
                              $("#msg-status").html("<div class='alert alert-info' role='alert'> "+response['message']+"<a class='close'  onclick=\"$('.alert').fadeOut(1000)\">&times;</a></div>")
                                $("#predict_from_model").prop('disabled', false);
                                      }
                     },
                     error: function(error)
                     {
                      $('#execution_id').val(response['execution_id']);
                         $("#msg-status").html("<div class='alert alert-danger' role='alert'> "+response['message']+"<a class='close'  onclick=\"$('.alert').fadeOut(1000)\">&times;</a></div>")
                            $("#predict_from_model").prop('disabled', false);

                     }


        });


    });
});

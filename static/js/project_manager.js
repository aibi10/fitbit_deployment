$(document).ready(function(){

//whenever input box get focus error message should be gone
    $("#project_name").focus(function(){
        $("#project-name-msg").html("");
    });
    $("#project_description").focus(function(){
        $("#project-des-msg").html("");
    });

    //call when project need to save
    $("#save-project").click(function(){
    $("#save-project").prop('disabled', true);
        var project_name= $("#project_name").val();
        var project_description=$("#project_description").val();
        var should_cancel=false;
        if (project_name.length<=0)
        {
            should_cancel=true;
            $("#project-name-msg").html("Please enter project name");
        }
        if (project_description.length<=0)
        {
            should_cancel=true
            $("#project-des-msg").html("Please enter project description");
        }
        if(should_cancel)
        {
         $("#msg-status").html("");
        $("#save-project").prop('disabled', false);
            return false;
        }
         $("#msg-status").html("<div class='alert alert-info'><i style='color:#008cba;' class='fas fa-sync fa-spin fa-1x'></i></div>");

         $.ajax({
                     url:'/save_project',
                     type: 'POST',
                     dataType: 'json',
                     data:  JSON.stringify({
                                        project_name:project_name,
                                        project_description:project_description

                                    }),
                     contentType: 'application/json',

                     success: function(response)
                     {

                        if(response['status']==true)
                        {
                            $("#msg-status").html("<div class='alert alert-success' role='alert'> "+response['message']+"<a class='close'  onclick=\"$('.alert').fadeOut(1000)\">&times;</a></div>")
                            $("#project_name").val("");
                            $("#project_description").val("");

                             $("#save-project").prop('disabled', false);

                             $("#project-list").append(`

                             <div class='col-md-3'>
                                    <div class='panel panel-default card' style="height:30%;">
                                        <div class='panel-heading'>
                                            <h3 class='panel-title'>`+project_name+`</h3></div>
                                        <div class='panel-body'><span>`+project_description.substr(0, 100)+`</span>

                                        </div>
                                    </div>
                                </div>`)
                        }
                        else
                        {
                            $("#msg-status").html("<div class='alert alert-danger' role='alert'> "+response['message']+"<a class='close'  onclick=\"$('.alert').fadeOut(1000)\">&times;</a></div>")
                                $("#save-project").prop('disabled', false);
                        }
                     },
                     error: function(error)
                     {
                         $("#msg-status").html("<div class='alert alert-danger' role='alert'> "+response['message']+"<a class='close'  onclick=\"$('.alert').fadeOut(1000)\">&times;</a></div>")
                           $("#save-project").prop('disabled', false);
                     }
               });

    });
});


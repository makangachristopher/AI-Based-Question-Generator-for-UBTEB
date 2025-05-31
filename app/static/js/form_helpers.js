// Dynamic form population for program-course-paper hierarchy
document.addEventListener('DOMContentLoaded', function() {
    // Check if we're on a page with the program/course/paper select fields
    const programSelect = document.getElementById('program_id');
    const courseSelect = document.getElementById('course_id');
    const paperSelect = document.getElementById('paper_id');
    
    if (programSelect && courseSelect) {
        // When program selection changes, update courses dropdown
        programSelect.addEventListener('change', function() {
            const programId = this.value;
            
            // Reset course and paper dropdowns if "No Program" is selected
            if (programId == 0) {
                resetSelect(courseSelect, 'Select Program First');
                if (paperSelect) {
                    resetSelect(paperSelect, 'Select Course First');
                }
                return;
            }
            
            // Fetch courses for the selected program
            fetch(`/api/get_courses/${programId}`)
                .then(response => response.json())
                .then(data => {
                    // Clear existing options
                    courseSelect.innerHTML = '';
                    
                    // Add default option
                    const defaultOption = document.createElement('option');
                    defaultOption.value = 0;
                    defaultOption.textContent = 'Select Course';
                    courseSelect.appendChild(defaultOption);
                    
                    // Add courses from API
                    data.forEach(course => {
                        const option = document.createElement('option');
                        option.value = course.id;
                        option.textContent = course.name;
                        courseSelect.appendChild(option);
                    });
                    
                    // Reset paper dropdown if it exists
                    if (paperSelect) {
                        resetSelect(paperSelect, 'Select Course First');
                    }
                })
                .catch(error => {
                    console.error('Error fetching courses:', error);
                });
        });
    }
    
    if (courseSelect && paperSelect) {
        // When course selection changes, update papers dropdown
        courseSelect.addEventListener('change', function() {
            const courseId = this.value;
            
            // Reset paper dropdown if "No Course" is selected
            if (courseId == 0) {
                resetSelect(paperSelect, 'Select Course First');
                return;
            }
            
            // Fetch papers for the selected course
            fetch(`/api/get_papers/${courseId}`)
                .then(response => response.json())
                .then(data => {
                    // Clear existing options
                    paperSelect.innerHTML = '';
                    
                    // Add default option
                    const defaultOption = document.createElement('option');
                    defaultOption.value = 0;
                    defaultOption.textContent = 'Select Paper';
                    paperSelect.appendChild(defaultOption);
                    
                    // Add papers from API
                    data.forEach(paper => {
                        const option = document.createElement('option');
                        option.value = paper.id;
                        option.textContent = `${paper.name} (${paper.paper_code})`;
                        paperSelect.appendChild(option);
                    });
                })
                .catch(error => {
                    console.error('Error fetching papers:', error);
                });
        });
    }
    
    // Helper function to reset a select element with a default option
    function resetSelect(selectElement, defaultText) {
        selectElement.innerHTML = '';
        const defaultOption = document.createElement('option');
        defaultOption.value = 0;
        defaultOption.textContent = defaultText;
        selectElement.appendChild(defaultOption);
    }
});

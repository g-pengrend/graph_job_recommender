from dataclasses import dataclass
from typing import List, Optional, Tuple
from geopy.geocoders import Nominatim
from loguru import logger

from user_preference import UserPreferences

# Initialize geocoder
geolocator = Nominatim(user_agent="job_recommender_v1")

# Predefined locations
LOCATIONS = {
    "1": ("Singapore", (1.3521, 103.8198)),
    "2": ("Jurong East", (1.3329, 103.7436)), 
    "3": ("Tampines", (1.3496, 103.9568)),
    "4": ("Woodlands", (1.4382, 103.7891)),
    "5": ("Punggol", (1.4041, 103.9025)),
    # Add more locations as needed
}

# Importance levels
IMPORTANCE_LEVELS = [
    "Very important",
    "Important",
    "Not important", 
    "Not important at all"
]

# Predefined job preferences
JOB_PREFERENCES = {
    "1": {
        "title": "Software Engineer",
        "description": """
        Responsibilities:
        - Develop and maintain software applications
        - Debug and troubleshoot technical issues
        - Collaborate with cross-functional teams
        - Write clean, maintainable code
        
        Qualifications:
        - Bachelor's degree in Computer Science or related field
        - 3+ years software development experience
        - Strong problem-solving abilities
        - Excellent communication skills
        
        Skills:
        Technical:
        - Python programming
        - Machine learning frameworks
        - Web development (Full-stack)
        - Cloud platforms (AWS/GCP)
        - Version control (Git)
        
        Non-technical:
        - Team collaboration
        - Project management
        - Technical documentation
        - Time management
        """,
        "title_importance": "Very important",
        "description_importance": "Important"
    },
    "2": {
        "title": "Data Scientist",
        "description": """
        Responsibilities:
        - Analyze complex data sets
        - Build predictive models
        - Present insights to stakeholders
        - Develop data pipelines
        
        Qualifications:
        - Master's degree in Data Science, Statistics or related field
        - 2+ years experience in data science
        - Research background preferred
        - Published work is a plus
        
        Skills:
        Technical:
        - Statistical analysis
        - Deep learning frameworks
        - Natural language processing
        - Big data technologies
        - Python/R programming
        
        Non-technical:
        - Critical thinking
        - Research methodology
        - Business acumen
        - Communication skills
        """,
        "title_importance": "Important",
        "description_importance": "Very important"
    },
    "3": {
        "title": "Frontend Developer",
        "description": """
        Responsibilities:
        - Build responsive web interfaces
        - Optimize application performance
        - Implement UI/UX designs
        - Maintain frontend codebase
        
        Qualifications:
        - Bachelor's degree in Computer Science or related field
        - 2+ years frontend development experience
        - Portfolio of web projects
        - UI/UX knowledge
        
        Skills:
        Technical:
        - React/Vue.js frameworks
        - HTML5/CSS3
        - JavaScript/TypeScript
        - Responsive design
        - Version control
        
        Non-technical:
        - Design thinking
        - Attention to detail
        - User empathy
        - Team collaboration
        """,
        "title_importance": "Very important",
        "description_importance": "Important"
    },
    "4": {
        "title": "DevOps Engineer",
        "description": """
        Responsibilities:
        - Manage CI/CD pipelines
        - Maintain cloud infrastructure
        - Implement security measures
        - Monitor system performance
        
        Qualifications:
        - Bachelor's degree in Computer Science or related field
        - 4+ years DevOps experience
        - Cloud certification preferred
        - Security knowledge
        
        Skills:
        Technical:
        - CI/CD tools
        - Docker/Kubernetes
        - Infrastructure as Code
        - Cloud platforms (AWS/Azure)
        - Shell scripting
        
        Non-technical:
        - Problem-solving
        - System thinking
        - Documentation
        - Incident management
        """,
        "title_importance": "Important",
        "description_importance": "Very important"
    },
    "5": {
        "title": "Marketing Manager",
        "description": """
        Responsibilities:
        - Develop marketing strategies
        - Manage campaigns
        - Lead marketing team
        - Track campaign performance
        
        Qualifications:
        - Bachelor's degree in Marketing or related field
        - 5+ years marketing experience
        - Team management experience
        - Budget management skills
        
        Skills:
        Technical:
        - Digital marketing tools
        - Analytics platforms
        - CRM systems
        - Social media management
        - SEO/SEM
        
        Non-technical:
        - Strategic thinking
        - Leadership
        - Creative direction
        - Project management
        """,
        "title_importance": "Important",
        "description_importance": "Important"
    },
    "6": {
        "title": "Financial Analyst",
        "description": """
        Responsibilities:
        - Conduct financial analysis
        - Create financial models
        - Assess investment opportunities
        - Generate reports
        
        Qualifications:
        - Bachelor's degree in Finance or related field
        - CFA certification preferred
        - 3+ years financial analysis experience
        - Strong analytical background
        
        Skills:
        Technical:
        - Financial modeling
        - Bloomberg Terminal
        - Excel advanced functions
        - Financial software
        - Statistical analysis
        
        Non-technical:
        - Analytical thinking
        - Attention to detail
        - Report writing
        - Risk assessment
        """,
        "title_importance": "Very important",
        "description_importance": "Important"
    },
    "7": {
        "title": "Human Resources Manager",
        "description": """
        Responsibilities:
        - Oversee recruitment process
        - Manage employee relations
        - Develop HR policies
        - Lead training programs
        
        Qualifications:
        - Bachelor's degree in HR or related field
        - PHR/SPHR certification preferred
        - 5+ years HR experience
        - Management experience
        
        Skills:
        Technical:
        - HRIS systems
        - Payroll software
        - Applicant tracking systems
        - Performance management tools
        
        Non-technical:
        - People management
        - Conflict resolution
        - Communication
        - Policy development
        """,
        "title_importance": "Important",
        "description_importance": "Very important"
    },
    "8": {
        "title": "Healthcare Administrator",
        "description": """
        Responsibilities:
        - Manage healthcare operations
        - Ensure regulatory compliance
        - Coordinate patient care
        - Oversee staff management
        
        Qualifications:
        - Master's in Healthcare Administration
        - Healthcare certifications
        - 5+ years healthcare experience
        - Management background
        
        Skills:
        Technical:
        - Healthcare software
        - EMR systems
        - Medical billing
        - Compliance tools
        
        Non-technical:
        - Leadership
        - Patient care
        - Staff coordination
        - Policy implementation
        """,
        "title_importance": "Important",
        "description_importance": "Very important"
    },
    "9": {
        "title": "Civil Engineer",
        "description": """
        Responsibilities:
        - Design infrastructure projects
        - Oversee construction
        - Ensure safety compliance
        - Manage project teams
        
        Qualifications:
        - Bachelor's in Civil Engineering
        - PE license
        - 5+ years engineering experience
        - Project management background
        
        Skills:
        Technical:
        - AutoCAD/BIM
        - Structural analysis
        - Construction methods
        - Safety protocols
        
        Non-technical:
        - Project management
        - Team leadership
        - Problem-solving
        - Client communication
        """,
        "title_importance": "Very important",
        "description_importance": "Important"
    },
    "10": {
        "title": "Chef de Cuisine",
        "description": """
        Responsibilities:
        - Lead kitchen operations
        - Create menus
        - Manage food costs
        - Train kitchen staff
        
        Qualifications:
        - Culinary degree
        - 7+ years culinary experience
        - Food safety certification
        - Management experience
        
        Skills:
        Technical:
        - Culinary techniques
        - Menu planning
        - Food cost control
        - Kitchen equipment
        
        Non-technical:
        - Team leadership
        - Time management
        - Creativity
        - Quality control
        """,
        "title_importance": "Important",
        "description_importance": "Very important"
    },
    "11": {
        "title": "Environmental Scientist",
        "description": """
        Responsibilities:
        - Conduct environmental studies
        - Analyze environmental data
        - Write assessment reports
        - Develop conservation plans
        
        Qualifications:
        - Master's in Environmental Science
        - Research experience
        - Field work experience
        - Technical writing skills
        
        Skills:
        Technical:
        - Environmental testing
        - Data analysis
        - GIS software
        - Research methods
        
        Non-technical:
        - Scientific writing
        - Project management
        - Problem-solving
        - Communication
        """,
        "title_importance": "Very important",
        "description_importance": "Important"
    },
    "12": {
        "title": "Supply Chain Manager",
        "description": """
        Responsibilities:
        - Optimize supply chain
        - Manage inventory
        - Coordinate with vendors
        - Improve processes
        
        Qualifications:
        - Bachelor's in Supply Chain Management
        - APICS certification preferred
        - 5+ years supply chain experience
        - Management background
        
        Skills:
        Technical:
        - ERP systems
        - Inventory management
        - Logistics software
        - Data analysis
        
        Non-technical:
        - Vendor management
        - Strategic planning
        - Team leadership
        - Problem-solving
        """,
        "title_importance": "Important",
        "description_importance": "Very important"
    },
    "13": {
        "title": "Graphic Designer",
        "description": """
        Responsibilities:
        - Create visual designs
        - Develop brand assets
        - Produce digital content
        - Collaborate with clients
        
        Qualifications:
        - Bachelor's in Graphic Design
        - Portfolio required
        - 3+ years design experience
        - Brand design experience
        
        Skills:
        Technical:
        - Adobe Creative Suite
        - Digital illustration
        - Motion graphics
        - Web design
        
        Non-technical:
        - Creativity
        - Time management
        - Client communication
        - Project coordination
        """,
        "title_importance": "Very important",
        "description_importance": "Important"
    },
    "14": {
        "title": "Sales Director",
        "description": """
        Responsibilities:
        - Drive sales strategy
        - Lead sales team
        - Manage key accounts
        - Achieve revenue targets
        
        Qualifications:
        - Bachelor's degree required
        - 8+ years sales experience
        - Management experience
        - Industry knowledge
        
        Skills:
        Technical:
        - CRM systems
        - Sales analytics
        - Presentation tools
        - Market analysis
        
        Non-technical:
        - Leadership
        - Negotiation
        - Strategic planning
        - Relationship building
        """,
        "title_importance": "Important",
        "description_importance": "Very important"
    },
    "15": {
        "title": "Research Scientist",
        "description": """
        Responsibilities:
        - Design experiments
        - Conduct research
        - Analyze results
        - Publish findings
        
        Qualifications:
        - Ph.D. in relevant field
        - Research experience
        - Publication history
        - Grant writing experience
        
        Skills:
        Technical:
        - Laboratory techniques
        - Research methods
        - Data analysis
        - Scientific software
        
        Non-technical:
        - Critical thinking
        - Scientific writing
        - Project management
        - Collaboration
        """,
        "title_importance": "Very important",
        "description_importance": "Important"
    }
}

def display_available_jobs():
    """Display available preloaded job preferences"""
    logger.info("\nAvailable Job Profiles:")
    logger.info("=====================")
    for key, job in JOB_PREFERENCES.items():
        logger.info(f"{key}. {job['title']}")

def display_available_locations():
    """Display available preloaded locations"""
    logger.info("\nAvailable Locations:")
    logger.info("==================")
    for key, (name, _) in LOCATIONS.items():
        logger.info(f"{key}. {name}")

def display_importance_levels():
    """Display available importance levels"""
    logger.info("\nImportance Levels:")
    logger.info("================")
    for i, level in enumerate(IMPORTANCE_LEVELS, 1):
        logger.info(f"{i}. {level}")

def get_custom_location() -> Optional[Tuple[str, Tuple[float, float]]]:
    """Get custom location from user input"""
    location_input = input("\nEnter postal code or location name: ").strip()
    if not location_input:
        return None
        
    try:
        location_data = geolocator.geocode(location_input)
        if location_data:
            return (location_data.address, (location_data.latitude, location_data.longitude))
        else:
            logger.warning("Location not found")
            return None
    except Exception as e:
        logger.error(f"Error processing location: {e}")
        return None

def get_preloaded_preferences() -> UserPreferences:
    """Get preloaded preferences with user selection"""
    while True:
        # Display and select job profile
        display_available_jobs()
        job_choice = input("\nSelect a job profile (1-15): ").strip()
        
        if job_choice not in JOB_PREFERENCES:
            logger.warning("Invalid choice. Please select a number between 1-15.")
            continue
            
        job_pref = JOB_PREFERENCES[job_choice]

        # Get importance levels
        display_importance_levels()
        while True:
            try:
                title_imp = int(input("\nSelect title importance (1-4): ").strip())
                desc_imp = int(input("Select description importance (1-4): ").strip())
                if 1 <= title_imp <= 4 and 1 <= desc_imp <= 4:
                    job_pref["title_importance"] = IMPORTANCE_LEVELS[title_imp-1]
                    job_pref["description_importance"] = IMPORTANCE_LEVELS[desc_imp-1]
                    break
                else:
                    logger.warning("Please select numbers between 1-4")
            except ValueError:
                logger.warning("Please enter valid numbers")
        
        # Display and select location
        logger.info("\nLocation Options:")
        logger.info("1. Choose from preset locations")
        logger.info("2. Enter custom location")
        logger.info("3. Skip location")
        
        loc_option = input("\nSelect location option (1-3): ").strip()
        
        location_name = None
        location_coords = None
        
        if loc_option == "1":
            display_available_locations()
            loc_choice = input("\nSelect a location (1-5): ").strip()
            if loc_choice in LOCATIONS:
                location_name, location_coords = LOCATIONS[loc_choice]
            else:
                logger.warning("Invalid location choice, proceeding without location")
        elif loc_option == "2":
            custom_location = get_custom_location()
            if custom_location:
                location_name, location_coords = custom_location
        elif loc_option != "3":
            logger.warning("Invalid option, proceeding without location")
            
        # Get max distance if location is selected
        max_distance_km = 10.0
        if location_coords:
            try:
                max_distance_km = float(input("\nMaximum distance in kilometers (default: 10): ") or 10)
            except ValueError:
                logger.warning("Invalid distance value, using default: 10km")
                max_distance_km = 10.0
        
        return UserPreferences(
            location=location_coords,
            location_name=location_name,
            job_title=job_pref["title"],
            job_description=job_pref["description"],
            max_distance_km=max_distance_km,
            title_importance=job_pref["title_importance"],
            description_importance=job_pref["description_importance"]
        )